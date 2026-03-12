# Google Image Search for Computer Vision Lecture

This directory contains scripts for searching and downloading images using the Google Custom Search API. These scripts are designed to help find appropriate images for the Computer Vision lecture.

## Overview

There are two main scripts:

1. `google_image_search.py` - A standalone script for searching and downloading images
2. `google_image_search_mcp.py` - An MCP server that exposes the image search functionality as tools for LLM agents

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up a Google Custom Search Engine:
   - Go to [Google Programmable Search Engine](https://programmablesearch.google.com/create)
   - Create a new search engine
   - Enable "Image search" in the settings
   - Get your Search Engine ID (cx)

3. Update the `SEARCH_ENGINE_ID` constant in `google_image_search.py` with your Search Engine ID.

## Using the Standalone Script

The `google_image_search.py` script can be used directly from the command line:

```bash
# Search for images and download them
python google_image_search.py "chest x-ray with nodule" --num 5 --output lectures/08/media/downloads

# Specify image type and size
python google_image_search.py "pillow python logo" --type photo --size large
```

### Command Line Arguments

- `query` (required): The search query
- `--num`: Number of images to search for and download (default: 5)
- `--output`: Directory to save downloaded images (default: lectures/08/media/downloads)
- `--type`: Type of image to search for (clipart, face, lineart, stock, photo, animated)
- `--size`: Size of image to search for (huge, icon, large, medium, small, xlarge, xxlarge)
- `--credentials`: Path to credentials JSON file (default: gen-lang-client-0306464806-ee14603523e3.json)
- `--engine-id`: Custom Search Engine ID

### Using as a Python Module

You can also import and use the script as a Python module:

```python
from google_image_search import GoogleImageSearch

# Initialize the search client
search_client = GoogleImageSearch("gen-lang-client-0306464806-ee14603523e3.json", "YOUR_SEARCH_ENGINE_ID")

# Search for images
results = search_client.search_images("chest x-ray with nodule", num_results=5)

# Download an image
path = search_client.download_image(
    "https://example.com/image.jpg", 
    "lectures/08/media/downloads", 
    "xray_nodule_example.png"
)

# Search and download in one step
paths = search_client.search_and_download(
    "chest x-ray with nodule", 
    "lectures/08/media/downloads", 
    num_results=5
)
```

## Using the MCP Server

The `google_image_search_mcp.py` script creates an MCP server that exposes the image search functionality as tools for LLM agents.

### Starting the MCP Server

```bash
python google_image_search_mcp.py
```

### Adding to MCP Settings

To add the MCP server to your MCP settings, add the following to your MCP settings file:

```json
{
  "mcpServers": {
    "google-image-search": {
      "command": "python",
      "args": ["/path/to/google_image_search_mcp.py"],
      "env": {}
    }
  }
}
```

### Available MCP Tools

The MCP server exposes the following tools:

1. `search_images` - Search for images using Google Image Search
2. `download_image` - Download an image to a local directory
3. `analyze_images` - Analyze image search results to find the most relevant ones
4. `search_and_download` - Search for images and download the most relevant ones

## Example Workflow for Computer Vision Lecture

1. Identify FIXME tags in the lecture document:
   ```
   <!-- #FIXME: Add image: Pillow logo. lectures/08/media/pillow_logo.png -->
   ```

2. Search for an appropriate image:
   ```bash
   python google_image_search.py "Pillow Python library logo" --num 5
   ```

3. Review the downloaded images in `lectures/08/media/downloads`

4. Choose the best image and rename it to match the target path:
   ```bash
   cp lectures/08/media/downloads/pillow_python_library_logo_1.jpg lectures/08/media/pillow_logo.png
   ```

5. Update the FIXME tag in the lecture document:
   ```
   <!-- #FIXME: Added candidate image: Pillow logo. lectures/08/media/pillow_logo.png -->
   ```

## Using with LLM Agents

When the MCP server is running and connected, LLM agents can use the tools to search for and download images. For example:

```
User: Find me an image of a chest X-ray with a nodule.

LLM Agent: I'll search for an image of a chest X-ray with a nodule.

[Uses search_images tool]
[Reviews results]
[Uses download_image tool to download the best match]

LLM Agent: I've found and downloaded an image of a chest X-ray with a nodule. You can find it at lectures/08/media/downloads/chest_xray_nodule.jpg.
```

## Troubleshooting

- **API Quota Exceeded**: The Google Custom Search API has a daily quota. If you exceed it, you'll need to wait until the next day or use a different API key.
- **No Images Found**: Try different search terms or remove filters like image type and size.
- **Invalid Images**: Some URLs might not point to valid images. The script will skip these and try to download others.