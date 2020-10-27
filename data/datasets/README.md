This folder contains the datasets used for X-Class.
Due to size constraints, we uploaded the data to google drive, 
[here](https://drive.google.com/drive/folders/1w0g3c0z9eoV-IYHCcA54tBKiNTYJy-3J?usp=sharing)
is the download link. After download, you can unzip the zipped dataset through `unzip -o`.

## Data format
We also describe the dataset format for potential use of new datasets.  
All files should be placed in a folder with the dataset's name, in this directory. The files to
include are
- dataset.txt 
    - A text file containing documents, one per line. We will use BERT's tokenizer for tokenization.
- classes.txt
    - A text file containing the class names, one per line.
- labels.txt
    - A text file containing the class (index) of each document in `dataset.txt`, one label per line.
All the files should have the exact same names.
