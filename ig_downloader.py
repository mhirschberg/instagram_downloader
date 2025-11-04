#!/usr/bin/env python3
"""
Instagram Video Downloader using Bright Data Scraping Browser
Downloads videos from Instagram posts via Scraping Browser with JSON/GraphQL extraction
"""

import os
import sys
import asyncio
import time
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import re
import json
from urllib.parse import unquote

load_dotenv()

def setup_logger():
    logger = logging.getLogger("InstagramDownloader")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - Instagram Downloader - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:80]

class InstagramDownloaderSBR:
    """Instagram downloader using Scraping Browser with JSON/GraphQL extraction"""
    
    def __init__(self):
        self.logger = setup_logger()
        
        # Get credentials from .env
        sbr_username = os.getenv('SBR_USERNAME')
        sbr_password = os.getenv('SBR_PASSWORD')
        sbr_country = os.getenv('SBR_COUNTRY', 'de')
        
        if not sbr_username or not sbr_password:
            raise ValueError("SBR_USERNAME and SBR_PASSWORD must be set in .env file")
        
        # Scraping Browser endpoint
        self.sbr_endpoint = f'wss://{sbr_username}:{sbr_password}@brd.superproxy.io:9222'
        
        # Settings
        self.output_dir = "downloads"
        self.timeout = 90000  # 90 seconds in milliseconds
        self.max_concurrent = 8  # Reasonable concurrency for browser instances
        
        # Stats
        self.stats_lock = threading.Lock()
        self.stats = {
            'successful': 0, 'failed': 0, 'total': 0, 'processed': 0,
            'browser_instances': 0, 'navigation_errors': 0, 'download_errors': 0,
            'video_not_found': 0, 'post_not_available': 0
        }
        
        create_directory(self.output_dir)
        self.start_time = time.time()
        
        self.logger.info(f"Instagram Downloader initialized with Scraping Browser")
        self.logger.info(f"Connection: {self.sbr_endpoint.split('@')[1]}")
        self.logger.info(f"Max concurrent browsers: {self.max_concurrent}")
        self.logger.info(f"Strategy: JSON/GraphQL extraction with DASH manifest support")
    
    def _update_stats(self, key: str, value=1):
        with self.stats_lock:
            if isinstance(value, int) and key in self.stats:
                self.stats[key] += value
            else:
                self.stats[key] = value
    
    def _get_current_stats(self) -> Dict:
        with self.stats_lock:
            return self.stats.copy()
    
    def extract_video_url_from_json(self, html_content: str, video_index: int) -> Optional[str]:
        """Extract video URL from Instagram's embedded JSON data"""
        
        # Method 1: Look for xdt_api__v1__media__shortcode__web_info
        json_pattern = r'"xdt_api__v1__media__shortcode__web_info":\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
        
        match = re.search(json_pattern, html_content)
        if match:
            try:
                json_str = match.group(1)
                # Clean up JSON escaping
                json_str = json_str.replace('\\u0026', '&').replace('\\/', '/')
                
                data = json.loads(json_str)
                
                # Navigate the JSON structure
                items = data.get('items', [])
                if not items:
                    self.logger.debug(f"[{video_index+1}] No items in web_info JSON")
                else:
                    item = items[0]
                    
                    # Try video_versions first (direct MP4 URLs)
                    video_versions = item.get('video_versions', [])
                    if video_versions:
                        url = video_versions[0]['url']
                        self.logger.info(f"[{video_index+1}] ✓ Found video URL via video_versions in JSON")
                        return url.replace('&amp;', '&')
                    
                    # Try DASH manifest
                    dash_manifest = item.get('video_dash_manifest', '')
                    if dash_manifest:
                        self.logger.info(f"[{video_index+1}] Found DASH manifest, extracting video URL...")
                        
                        # Unescape the manifest
                        manifest = dash_manifest.replace('\\n', '\n').replace('\\/', '/').replace('&amp;', '&')
                        
                        # Extract all BaseURL entries
                        base_url_pattern = r'<BaseURL>([^<]+)</BaseURL>'
                        urls = re.findall(base_url_pattern, manifest)
                        
                        # Filter for video URLs (not audio)
                        video_urls = []
                        for url in urls:
                            url_clean = url.replace('&amp;', '&')
                            # Check if it's a video URL (contains video codecs or higher resolution)
                            if any(indicator in url_clean.lower() for indicator in ['avc1', 'h264', 'video']):
                                video_urls.append(url_clean)
                            # Skip audio-only URLs
                            elif 'audio' in url_clean.lower() and 'video' not in url_clean.lower():
                                continue
                            # If no clear indicator, include .mp4 URLs
                            elif '.mp4' in url_clean:
                                video_urls.append(url_clean)
                        
                        if video_urls:
                            # Return the first video URL (usually highest quality)
                            self.logger.info(f"[{video_index+1}] ✓ Found video URL via DASH manifest")
                            self.logger.info(f"[{video_index+1}] Available qualities: {len(video_urls)}")
                            return video_urls[0]
                        
            except json.JSONDecodeError as e:
                self.logger.debug(f"[{video_index+1}] JSON parsing error: {str(e)}")
            except Exception as e:
                self.logger.debug(f"[{video_index+1}] Error extracting from JSON: {str(e)}")
        
        # Method 2: Look for video_versions pattern directly in HTML
        video_versions_patterns = [
            r'"video_versions":\s*\[\s*\{\s*"[^"]*":\s*[^,]+,\s*"[^"]*":\s*[^,]+,\s*"url":\s*"([^"]+)"',
            r'"video_versions":\[{"[^"]*":[^,]*,"[^"]*":[^,]*,"url":"([^"]+)"',
        ]
        
        for pattern in video_versions_patterns:
            matches = re.findall(pattern, html_content)
            if matches:
                url = matches[0].replace('\\u0026', '&').replace('\\/', '/').replace('&amp;', '&')
                self.logger.info(f"[{video_index+1}] ✓ Found video URL via video_versions pattern")
                return url
        
        # Method 3: Look for video_url or playback_url
        direct_url_patterns = [
            r'"video_url":\s*"(https://[^"]+\.mp4[^"]*)"',
            r'"playback_url":\s*"(https://[^"]+\.mp4[^"]*)"',
            r'"src":\s*"(https://[^"]+\.mp4[^"]*)"',
            r'videoUrl&quot;:&quot;(https://[^&]+?)&quot;',
        ]
        
        for pattern in direct_url_patterns:
            matches = re.findall(pattern, html_content)
            if matches:
                url = matches[0].replace('\\u0026', '&').replace('\\/', '/').replace('&amp;', '&')
                self.logger.info(f"[{video_index+1}] ✓ Found video URL via direct pattern")
                return url
        
        return None
    
    async def download_single_video_fresh_browser(self, item: Dict, video_index: int) -> Tuple[bool, str]:
        """Download video using fresh browser instance"""
        
        playwright = None
        browser = None
        context = None
        page = None
        
        instagram_url = item.get('url')
        user_posted = item.get('user_posted', 'unknown')
        shortcode = item.get('shortcode', 'unknown')
        content_id = item.get('content_id', 'unknown')
        
        self.logger.info(f"[{video_index+1}] Starting: {user_posted}/{shortcode}")
        self.logger.info(f"[{video_index+1}] URL: {instagram_url}")
        
        try:
            # Create fresh playwright instance
            self.logger.info(f"[{video_index+1}] Creating fresh browser instance...")
            playwright = await async_playwright().start()
            
            # Connect to Scraping Browser
            self.logger.info(f"[{video_index+1}] Connecting to Scraping Browser...")
            browser = await playwright.chromium.connect_over_cdp(self.sbr_endpoint)
            self._update_stats('browser_instances')
            self.logger.info(f"[{video_index+1}] ✓ Browser connected")
            
            # Create context and page
            self.logger.info(f"[{video_index+1}] Creating browser context...")
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                ignore_https_errors=True
            )
            
            page = await context.new_page()
            page.set_default_timeout(self.timeout)
            self.logger.info(f"[{video_index+1}] ✓ Page created")
            
            # Step 1: Navigate to Instagram page
            try:
                self.logger.info(f"[{video_index+1}] Navigating to Instagram post...")
                await page.goto(instagram_url, timeout=self.timeout, wait_until='networkidle')
                self.logger.info(f"[{video_index+1}] ✓ Page loaded")
                
                # Wait for dynamic content
                self.logger.info(f"[{video_index+1}] Waiting for dynamic content (8s)...")
                await page.wait_for_timeout(8000)
                
            except Exception as nav_error:
                self.logger.error(f"[{video_index+1}] ✗ Navigation failed: {str(nav_error)}")
                self._update_stats('navigation_errors')
                self._update_stats('failed')
                self._update_stats('processed')
                return False, f"Navigation failed: {str(nav_error)}"
            
            # Step 2: Check if post is available
            page_title = await page.title()
            self.logger.info(f"[{video_index+1}] Page title: {page_title}")
            
            # Check for error messages
            error_indicators = [
                "não está disponível",
                "not available",
                "corrompido",
                "removido",
                "Sorry, this page",
            ]
            
            if any(indicator.lower() in page_title.lower() for indicator in error_indicators):
                self.logger.error(f"[{video_index+1}] ✗ Post is not available")
                self._update_stats('post_not_available')
                self._update_stats('failed')
                self._update_stats('processed')
                return False, "Post not available (deleted, private, or requires login)"
            
            # Step 3: Extract page content
            self.logger.info(f"[{video_index+1}] Extracting page content...")
            html_content = await page.content()
            html_size = len(html_content)
            self.logger.info(f"[{video_index+1}] Page content size: {html_size} bytes")
            
            # Step 4: Extract video URL using JSON/GraphQL method
            self.logger.info(f"[{video_index+1}] Looking for video URL in page data...")
            
            final_video_url = self.extract_video_url_from_json(html_content, video_index)
            
            if not final_video_url:
                self.logger.error(f"[{video_index+1}] ✗ No video URL found in JSON data")
                self._update_stats('video_not_found')
                self._update_stats('failed')
                self._update_stats('processed')
                
                # Save debug HTML
                debug_file = Path(self.output_dir) / f"debug_{shortcode}.html"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"[{video_index+1}] Saved debug HTML to: {debug_file}")
                
                # Also save the JSON extraction attempt for debugging
                debug_json = Path(self.output_dir) / f"debug_{shortcode}_json.txt"
                with open(debug_json, 'w', encoding='utf-8') as f:
                    # Try to extract and save the JSON data
                    json_pattern = r'"xdt_api__v1__media__shortcode__web_info":\s*(\{.+?\})\s*(?=,"extensions")'
                    match = re.search(json_pattern, html_content, re.DOTALL)
                    if match:
                        f.write("Found JSON data:\n")
                        f.write(match.group(1)[:5000])  # First 5000 chars
                    else:
                        f.write("No JSON data found in expected location\n")
                        # Look for video_dash_manifest
                        dash_pattern = r'"video_dash_manifest":\s*"([^"]+)"'
                        dash_match = re.search(dash_pattern, html_content)
                        if dash_match:
                            f.write("\n\nFound DASH manifest:\n")
                            f.write(dash_match.group(1)[:5000])
                
                return False, f"No video URL found"
            
            # Clean up the URL
            final_video_url = unquote(final_video_url)
            self.logger.info(f"[{video_index+1}] Final video URL: {final_video_url[:150]}...")
            
            # Step 5: Create filename
            safe_username = sanitize_filename(user_posted)
            safe_shortcode = sanitize_filename(shortcode)
            filename = f"{safe_username}_{safe_shortcode}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            self.logger.info(f"[{video_index+1}] Target filename: {filename}")
            
            # Check if exists
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                self.logger.info(f"[{video_index+1}] ✓ File already exists, skipping download")
                self._update_stats('successful')
                self._update_stats('processed')
                return True, f"Already exists: {filename}"
            
            # Step 6: Download video using same browser context
            try:
                self.logger.info(f"[{video_index+1}] Downloading video...")
                
                video_response = await context.request.get(
                    final_video_url,
                    headers={
                        'Referer': 'https://www.instagram.com/',
                        'Origin': 'https://www.instagram.com',
                        'Accept': '*/*',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    },
                    timeout=90000  # 90 seconds for video download
                )
                
                self.logger.info(f"[{video_index+1}] Video response status: {video_response.status}")
                
                if video_response.status in [200, 206]:
                    video_data = await video_response.body()
                    data_size = len(video_data)
                    self.logger.info(f"[{video_index+1}] Downloaded {data_size} bytes ({data_size/1024/1024:.2f} MB)")
                    
                    if data_size > 10000:
                        self.logger.info(f"[{video_index+1}] Saving video to disk...")
                        with open(filepath, 'wb') as f:
                            f.write(video_data)
                        
                        file_size = os.path.getsize(filepath)
                        size_mb = file_size / (1024 * 1024)
                        
                        self.logger.info(f"[{video_index+1}] ✓ Successfully saved: {filename} ({size_mb:.1f}MB)")
                        
                        self._update_stats('successful')
                        self._update_stats('processed')
                        
                        return True, f"Success: {filename} ({size_mb:.1f}MB)"
                    else:
                        self.logger.error(f"[{video_index+1}] ✗ Video data too small: {data_size} bytes")
                        self._update_stats('download_errors')
                        self._update_stats('failed')
                        self._update_stats('processed')
                        return False, f"Video too small: {data_size}B"
                else:
                    self.logger.error(f"[{video_index+1}] ✗ Download failed with HTTP {video_response.status}")
                    self._update_stats('download_errors')
                    self._update_stats('failed')
                    self._update_stats('processed')
                    return False, f"Download failed: HTTP {video_response.status}"
                    
            except Exception as download_error:
                self.logger.error(f"[{video_index+1}] ✗ Download error: {str(download_error)}")
                self._update_stats('download_errors')
                self._update_stats('failed')
                self._update_stats('processed')
                return False, f"Download error: {str(download_error)}"
                
        except Exception as e:
            self.logger.error(f"[{video_index+1}] ✗ Browser error: {str(e)}")
            import traceback
            self.logger.error(f"[{video_index+1}] Traceback:\n{traceback.format_exc()}")
            self._update_stats('failed')
            self._update_stats('processed')
            return False, f"Browser error: {str(e)}"
        
        finally:
            # Always cleanup browser resources
            self.logger.info(f"[{video_index+1}] Cleaning up browser resources...")
            try:
                if page and not page.is_closed():
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()
                self.logger.info(f"[{video_index+1}] ✓ Cleanup complete")
            except Exception as cleanup_error:
                self.logger.debug(f"[{video_index+1}] Cleanup error: {str(cleanup_error)}")
    
    def run_async_download(self, item: Dict, video_index: int) -> Tuple[bool, str]:
        """Wrapper to run async download in sync context"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self.download_single_video_fresh_browser(item, video_index))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"[{video_index+1}] Async execution error: {str(e)}")
            self._update_stats('failed')
            self._update_stats('processed')
            return False, f"Async execution error: {str(e)}"
    
    def download_batch(self, items: List[Dict], batch_num: int) -> List[Dict]:
        """Download batch using fresh browsers"""
        results = []
        
        self.logger.info(f"Batch {batch_num}: Processing {len(items)} Instagram posts with fresh browsers")
        
        # Use ThreadPoolExecutor to run async operations
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_item = {}
            
            for i, item in enumerate(items):
                video_index = (batch_num - 1) * len(items) + i
                future = executor.submit(self.run_async_download, item, video_index)
                future_to_item[future] = (item, video_index)
            
            completed = 0
            for future in as_completed(future_to_item):
                item, video_index = future_to_item[future]
                url = item.get('url', 'N/A')
                
                try:
                    success, message = future.result(timeout=120)  # 2 minutes per video
                    
                    # Log the result
                    if success:
                        self.logger.info(f"[{video_index+1}] ✓ RESULT: {message}")
                    else:
                        self.logger.error(f"[{video_index+1}] ✗ RESULT: {message}")
                    
                    results.append({
                        'url': url,
                        'success': success,
                        'message': message
                    })
                    
                    completed += 1
                    
                    # Log progress every 3 videos
                    if completed % 3 == 0 or completed == len(items):
                        current_stats = self._get_current_stats()
                        elapsed = time.time() - self.start_time
                        rate = current_stats['processed'] / elapsed * 60 if elapsed > 0 else 0
                        success_rate = (current_stats['successful'] / max(current_stats['processed'], 1)) * 100
                        
                        self.logger.info(
                            f"Batch {batch_num} progress: {completed}/{len(items)} completed | "
                            f"Overall: {current_stats['successful']}/{current_stats['processed']} success ({success_rate:.1f}%) | "
                            f"Rate: {rate:.1f} videos/min"
                        )
                    
                    # Delay between browser creations
                    time.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    self.logger.error(f"[{video_index+1}] ✗ Execution exception: {str(e)}")
                    results.append({
                        'url': url,
                        'success': False,
                        'message': f"Execution error: {str(e)}"
                    })
                    self._update_stats('failed')
                    self._update_stats('processed')
        
        return results
    
    def test_connection(self) -> bool:
        """Test Scraping Browser connection"""
        try:
            self.logger.info("Testing Scraping Browser connection")
            
            # Test with simple async function
            async def test_simple_connection():
                playwright = None
                browser = None
                try:
                    playwright = await async_playwright().start()
                    browser = await playwright.chromium.connect_over_cdp(self.sbr_endpoint)
                    
                    context = await browser.new_context()
                    page = await context.new_page()
                    
                    # Test simple navigation
                    await page.goto("https://example.com", timeout=30000, wait_until='domcontentloaded')
                    content = await page.content()
                    
                    await page.close()
                    await context.close()
                    
                    return len(content) > 100
                    
                except Exception as e:
                    self.logger.error(f"Connection test error: {str(e)}")
                    return False
                finally:
                    if browser:
                        try:
                            await browser.close()
                        except:
                            pass
                    if playwright:
                        try:
                            await playwright.stop()
                        except:
                            pass
            
            # Run test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(test_simple_connection())
                if result:
                    self.logger.info("✓ Scraping Browser connection test successful")
                    return True
                else:
                    self.logger.error("✗ Scraping Browser connection test failed")
                    return False
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def download_all_videos(self, items: List[Dict]) -> Dict:
        """Download all videos using fresh browsers"""
        self.stats['total'] = len(items)
        
        self.logger.info(f"Instagram Downloader starting: {len(items)} videos")
        self.logger.info(f"Using fresh Scraping Browser instance per video")
        self.logger.info(f"Concurrent browsers: {self.max_concurrent}")
        
        results = []
        batch_size = 20  # Smaller batches for browser operations
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(items) + batch_size - 1) // batch_size
            
            batch_results = self.download_batch(batch, batch_num)
            results.extend(batch_results)
            
            # Batch summary
            batch_successful = sum(1 for r in batch_results if r['success'])
            batch_success_rate = (batch_successful / len(batch)) * 100
            
            current_stats = self._get_current_stats()
            overall_success_rate = (current_stats['successful'] / max(current_stats['processed'], 1)) * 100
            
            self.logger.info(
                f"Batch {batch_num}/{total_batches} completed: {batch_successful}/{len(batch)} successful ({batch_success_rate:.1f}%)"
            )
            
            self.logger.info(
                f"Overall progress: {current_stats['successful']}/{current_stats['processed']} successful ({overall_success_rate:.1f}%)"
            )
            
            # Pause between batches
            if i + batch_size < len(items):
                pause_time = random.uniform(10, 20)
                self.logger.info(f"Pausing {pause_time:.1f}s between batches")
                time.sleep(pause_time)
        
        return {'results': results, 'stats': self.stats}

def is_url_list_file(file_path: str) -> bool:
    """Detect if file is a simple list of URLs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Check if first line is an Instagram URL
            return first_line.startswith('http') and ('instagram.com/p/' in first_line or 'instagram.com/reel/' in first_line)
    except:
        return False

def load_url_list(file_path: str) -> List[Dict]:
    """Load URLs from text file and convert to item format"""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            url = line.strip()
            if url and 'instagram.com' in url:
                # Extract shortcode from /p/ or /reel/ URLs
                if '/p/' in url:
                    shortcode = url.split('/p/')[1].split('/')[0]
                elif '/reel/' in url:
                    shortcode = url.split('/reel/')[1].split('/')[0]
                else:
                    shortcode = 'unknown'
                
                items.append({
                    'url': url,
                    'user_posted': 'unknown',
                    'shortcode': shortcode,
                    'content_id': 'unknown'
                })
    return items

def load_instagram_data(input_file: str) -> List[Dict]:
    """Load Instagram post data from JSON file or URL list"""
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    # Check if it's a URL list first
    if is_url_list_file(input_file):
        return load_url_list(input_file)
    
    # Try JSON parsing
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("File must be either JSON or a text file with Instagram URLs")

    # Handle both single object and array
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON must be an object or array of objects")
    
    # Validate items have URL field
    valid_items = []
    for i, item in enumerate(items):
        if not item.get('url'):
            print(f"Warning: Item {i+1} missing 'url' field, skipping")
            continue
        valid_items.append(item)
    
    return valid_items

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python ig_downloader.py <input_file>")
        print("Example: python ig_downloader.py one.json")
        print("         python ig_downloader.py urls.txt")
        print("\nSupported formats:")
        print("  - JSON file with Instagram post data")
        print("  - Text file with Instagram URLs (one per line)")
        print("\nMake sure .env file contains:")
        print("  SBR_USERNAME=brd-customer-hl_xxxxx-zone-scraping_browser")
        print("  SBR_PASSWORD=your_password_here")
        print("  SBR_COUNTRY=de")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load Instagram data
        items = load_instagram_data(input_file)
        
        print(f"\n{'='*60}")
        print(f"Instagram Video Downloader - JSON/GraphQL Extraction Mode")
        print(f"{'='*60}")
        print(f"Posts to process: {len(items)}")
        print(f"Strategy: Extract from Instagram's embedded JSON data")
        print(f"{'='*60}\n")
        
        # Initialize downloader
        downloader = InstagramDownloaderSBR()
        
        # Test connection
        if not downloader.test_connection():
            print("\n✗ Scraping Browser connection test failed")
            print("Please check your .env file credentials")
            sys.exit(1)
        
        # Download videos
        start_time = time.time()
        results = downloader.download_all_videos(items)
        duration = time.time() - start_time
        
        # Final results
        stats = results['stats']
        
        print(f"\n{'='*60}")
        print(f"Instagram Downloader Completed")
        print(f"{'='*60}")
        print(f"Results: {stats['successful']}/{stats['total']} successful ({(stats['successful']/max(stats['total'],1)*100):.1f}%)")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Rate: {(len(items)/(duration/60)):.1f} videos/min")
        print(f"Browser instances created: {stats['browser_instances']}")
        
        if stats['navigation_errors'] > 0:
            print(f"Navigation errors: {stats['navigation_errors']}")
        if stats['download_errors'] > 0:
            print(f"Download errors: {stats['download_errors']}")
        if stats.get('video_not_found', 0) > 0:
            print(f"Video not found errors: {stats['video_not_found']}")
        if stats.get('post_not_available', 0) > 0:
            print(f"Post not available errors: {stats['post_not_available']}")
        
        print(f"\nFiles saved to: {Path(downloader.output_dir).absolute()}")
        print(f"{'='*60}\n")
        
        # Show individual results
        if stats['failed'] > 0:
            print("\nFailed downloads:")
            for i, result in enumerate(results['results']):
                if not result['success']:
                    print(f"  {i+1}. {result['url']}")
                    print(f"     Error: {result['message']}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

