# image-style-transfer-perserving-color

## requirement:
  * tensorflow
  * numpy
  * scipy

## usage

**python main.py --content** *content image path* **--style** *style image path* **--output** *otput path* **--p** *the method of perserving color*

**--p**
**histogram_match** *color histogram match method*
**luminance** *luminance channel transfer method*
**None** *do nothing*
