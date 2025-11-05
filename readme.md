# Inverted Index Creation

A Python implementation of inverted index creation for text documents, supporting both unigram and bigram indexing.

## Overview

This project implements a MapReduce-style inverted index creator that processes text documents and generates:
- **Unigram Index**: Word frequency across all documents from fulldata (74 files)
- **Bigram Index**: Two-word phrase frequency for specific target bigrams from fulldata (74 files)

## Requirements

- Python 3.x
- Standard libraries: `os`, `sys`, `re`, `string`, `collections`

## Project Structure

```
inverted-index-creation/
├── main.py                              # Main implementation
├── readme.md                            # This file
├── data/
│   ├── devdata/                         # Development dataset (5 files) - used for testing
│   │   ├── 5722018101.txt
│   │   ├── 5722018235.txt
│   │   ├── 5722018301.txt
│   │   ├── 5722018496.txt
│   │   └── 5722018508.txt
│   └── fulldata/                        # Full dataset (74 files) - used for final indices
│       ├── 5722018435.txt
│       ├── 5722018436.txt
│       └── ... (72 more files)
└── output/                              # Generated output files
    ├── unigram_index.txt                # Generated unigram index from fulldata
    └── selected_bigram_index.txt        # Generated bigram index from fulldata
```

## Usage

### Simple Execution (Recommended)

```bash
python3 main.py
```

This single command will:
- Create an `output/` directory if it doesn't exist
- Process all 74 files from `data/fulldata/`
- Generate `output/unigram_index.txt` (~96MB, ~1.2M unique words)
- Generate `output/selected_bigram_index.txt` (~2KB, 5 target bigrams with comprehensive counts)
- Display progress and statistics
- Remind you to take screenshots of the output folder

### Test Mode

```bash
python3 main.py test
```

Runs test functions using the smaller devdata directory to verify functionality.

### Manual/Interactive Usage

#### 1. Interactive Python Session
```python
python3
>>> from main import *

>>> # Process unigrams from fulldata
>>> unigram_index, docs = process_directory('data/fulldata')
>>> write_inverted_index(unigram_index, 'output/unigram_index.txt')

>>> # Process bigrams from fulldata
>>> target_bigrams = {('computer', 'science'), ('information', 'retrieval'), 
...                   ('power', 'politics'), ('los', 'angeles'), ('bruce', 'willis')}
>>> bigram_index, docs = process_directory_bigrams('data/fulldata', target_bigrams)
>>> write_inverted_index(bigram_index, 'output/selected_bigram_index.txt')
```

#### 2. Command Line Execution
```bash
# Generate unigram index from fulldata (74 files)
python3 -c "
from main import *
unigram_index, docs = process_directory('data/fulldata')
write_inverted_index(unigram_index, 'unigram_index.txt')
print(f'Generated unigram index with {len(unigram_index)} unique words from {len(docs)} documents')
"

# Generate bigram index from fulldata (74 files)
python3 -c "
from main import *
target_bigrams = {('computer', 'science'), ('information', 'retrieval'), 
                  ('power', 'politics'), ('los', 'angeles'), ('bruce', 'willis')}
bigram_index, docs = process_directory_bigrams('data/fulldata', target_bigrams)
write_inverted_index(bigram_index, 'bigram_index.txt')
print(f'Generated bigram index for {len(bigram_index)} bigrams from {len(docs)} documents')
"
```

## Input Format

Documents must be in tab-separated format:
```
docID<TAB>document_content
```

Example:
```
5722018101	This is the content of document 5722018101 with various words and phrases.
```

## Output Format

Both unigram and bigram indices use the same format:
```
term<TAB>docID1:count1 docID2:count2 docID3:count3 ...
```

### Unigram Index Example
```
computer	5722018101:45 5722018235:23 5722018301:12
science	5722018101:34 5722018235:18 5722018508:67
```

### Bigram Index Example
```
computer science	5722018435:746 5722018436:136 5722018437:1622 5722018438:57 ...
information retrieval	5722018435:108 5722018436:8 5722018437:73 5722018438:15 ...
```

## Key Functions

### Core Processing Functions
- `parse_document_file(file_path)`: Parse tab-separated document files
- `clean_text(text)`: Remove punctuation/numerals, convert to lowercase
- `tokenize_text(text)`: Split cleaned text into words
- `generate_bigrams(words, target_bigrams)`: Generate target bigrams from word list

### Directory Processing
- `process_directory(directory)`: Process all documents for unigrams
- `process_directory_bigrams(directory, target_bigrams)`: Process documents for specific bigrams

### Output Functions
- `write_inverted_index(index, output_file)`: Write index to file in required format

## Text Processing Details

1. **Cleaning**: All punctuation and numerals replaced with spaces
2. **Normalization**: All text converted to lowercase
3. **Tokenization**: Text split on whitespace
4. **Bigram Generation**: Only generates specified target bigrams

## Target Bigrams

The system is configured to find these specific bigrams:
- "computer science"
- "information retrieval"
- "power politics"
- "los angeles"
- "bruce willis"

## Performance

- **Unigram Processing**: ~530M words, 1.2M unique terms (74 files)
- **Bigram Processing**: 5 target bigrams across 74 files
- **Output Size**: Unigram index ~96MB, Bigram index ~2KB

## Expected Results

### Unigram Index (`output/unigram_index.txt`)
- **Size**: ~96MB
- **Unique Words**: ~1,229,381
- **Documents**: 74 files from fulldata
- **Total Words Processed**: ~530 million

### Bigram Index (`output/selected_bigram_index.txt`) 
- **Size**: ~2KB
- **Target Bigrams Found**: 5
- **Documents**: 74 files from fulldata
- **Expected Counts**:
  - "computer science": ~18,027 occurrences
  - "information retrieval": ~1,926 occurrences  
  - "los angeles": ~14,997 occurrences
  - "power politics": ~167 occurrences
  - "bruce willis": ~53 occurrences

## Example Complete Workflow

```bash
# Single command to generate both indices
python3 main.py
```

**Expected Output:**
```
Created output directory: output
============================================================
INVERTED INDEX GENERATION
============================================================

1. Generating unigram index from fulldata (74 files)...
Found 74 .txt files in data/fulldata
Processing 1/74: 5722018435.txt...
  ✓ Processed 5722018435: 92875 unique words, 7594940 total words
Processing 2/74: 5722018436.txt...
  ✓ Processed 5722018436: 83125 unique words, 7636891 total words
...
   ✓ Generated output/unigram_index.txt
   ✓ Processed 74 documents
   ✓ Created index with 1229381 unique words

2. Generating bigram index from fulldata (74 files)...
Found 74 .txt files in data/fulldata
Processing 1/74: 5722018435.txt...
  ✓ Processed 5722018435: 3 target bigrams found, 960 total instances
...
   ✓ Generated output/selected_bigram_index.txt
   ✓ Processed 74 documents
   ✓ Found 5 target bigrams
   ✓ Bigram counts:
      'bruce willis': 53 occurrences
      'computer science': 18027 occurrences
      'information retrieval': 1926 occurrences
      'los angeles': 14997 occurrences
      'power politics': 167 occurrences

============================================================
GENERATION COMPLETE
============================================================

Submission files generated in 'output' folder:
  📄 output/unigram_index.txt - Unigram inverted index
  📄 output/selected_bigram_index.txt - Bigram inverted index
  📄 main.py - Source code

To run tests instead: python3 main.py test

📸 IMPORTANT: Take screenshots of the 'output' folder for homework submission!
```

### Manual Step-by-Step (Alternative)

```bash
# 1. Generate unigram index from full dataset
python3 -c "
from main import *
print('Processing unigrams from fulldata...')
unigram_index, docs = process_directory('data/fulldata')
write_inverted_index(unigram_index, 'unigram_index.txt')
print(f'✓ Generated unigram_index.txt: {len(unigram_index)} unique words')
"

# 2. Generate bigram index from full dataset  
python3 -c "
from main import *
print('Processing bigrams from fulldata...')
target_bigrams = {('computer', 'science'), ('information', 'retrieval'), 
                  ('power', 'politics'), ('los', 'angeles'), ('bruce', 'willis')}
bigram_index, docs = process_directory_bigrams('data/fulldata', target_bigrams)
write_inverted_index(bigram_index, 'bigram_index.txt')
print(f'✓ Generated bigram_index.txt: {len(bigram_index)} bigrams found')
"

# 3. Verify output files
ls -la output/*.txt
```

## Screenshots for Homework Submission

After running the script, you need to take screenshots of the output folder for your homework:

```bash
# Check the output folder contents
ls -la output/

# On macOS, open the output folder in Finder for screenshots
open output/

# On Linux/Windows, navigate to the output folder in your file manager
```

Take screenshots showing:
1. The `output` folder containing both `unigram_index.txt` and `selected_bigram_index.txt`
2. File sizes and timestamps to prove successful generation

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure data directories exist with proper file structure
2. **Memory Issues**: Large datasets may require sufficient RAM for processing
3. **Encoding Issues**: Files should be UTF-8 encoded

### Debug Mode
```python
# Enable verbose output
from main import *
unigram_index, docs = process_directory('data/devdata')  # Use smaller dataset
print(f"Processed {len(docs)} documents")
print(f"Found {len(unigram_index)} unique terms")
```

## Author

CS572 Information Retrieval Course Project