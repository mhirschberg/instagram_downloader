# Instagram Downloader
A Python script for downloading Instagram reels at scale using Bright Data's Scraping Browser API with Playwright automation.

## Features

- Downloads Instagram reels from a list of URLs
- Uses Bright Data Scraping Browser for advanced anti-detection
- Fresh browser instance per video to avoid navigation limits
- Real browser automation with JavaScript execution
- Adaptive concurrency based on performance
- Automatic retry and error handling
- Progress tracking and detailed reporting

## Requirements

- Python 3.7+
- Bright Data Scraping Browser zone
- Playwright and Chromium browser
- Optional: Bright Data's Instagram dataset snapshot

## Installation

1. Install required packages:
```bash
pip install playwright requests python-dotenv
```
2. Install Playwright browsers:
```bash
playwright install chromium
```
3. Create a `.env` file with your Bright Data Scraping Browser credentials:
```
SBR_USERNAME=brd-customer-hl_YOUR_CUSTOMER_ID-zone-YOUR_SBR_ZONE
SBR_PASSWORD=YOUR_SBR_PASSWORD
SBR_COUNTRY=de|us|...
```

## Usage
Create a file with Tiktok URLs, one per line:
```
https://www.instagram.com/reel/somereelurl1
https://www.instagram.com/reel/somereelurl2
```

Or create and download a Bright Data's Instagram reels snapshot with the filtering you need, save it as JSON file

## Run
Simply run the script with the URLs file as a parameter:
```
python ig_downloader.py urls.txt
```
or
```
python ig_downloader.py snapshot.json
```

## Output
The videos are saved in `downloads` directory
