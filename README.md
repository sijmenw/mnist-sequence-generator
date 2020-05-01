# MNIST Sequence Generator
Generates an image with multiple digits from MNIST data. Example use case would be to train an OCR system.

```
usage: mnist_sequence_generator.py [-h] [-w WIDTH] [-i MINMARGIN] [-a MAXMARGIN] [-l STRLEN]
                                   [-s NUMBERSTRING] [-n GENN] [-o OUTPUTDIR]

Generate images from MNIST images for OCR training purposes.

optional arguments:
  -h, --help            show this help message and exit
  -w WIDTH, --width WIDTH
                        Width of the resulting image
  -i MINMARGIN, --minmargin MINMARGIN
                        Minimum margin between MNIST characters
  -a MAXMARGIN, --maxmargin MAXMARGIN
                        Maximum margin between MNIST characters
  -l STRLEN, --strlen STRLEN
                        number of characters per string
  -s NUMBERSTRING, --numberstring NUMBERSTRING
                        string of numbers
  -n GENN, --genn GENN  number of images to generate
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        output directory for generated images
```

