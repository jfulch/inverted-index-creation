#!/usr/bin/env python3
"""
Generated with the help of GitHub Copilot and using Claude Sonnet 4
"""
import os
import sys
import re
import string
from collections import defaultdict, Counter

def parse_document_file(file_path):
    """
    Parse a document file and extract docID and content.
    
    Format: docID<TAB>content
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        tuple: (doc_id, content) or (None, None) if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire file content
            content = file.read().strip()
            
            # Split on the first tab character
            if '\t' in content:
                parts = content.split('\t', 1)  # Split only on first tab
                doc_id = parts[0].strip()
                document_content = parts[1].strip()
                return doc_id, document_content
            else:
                print(f"Warning: No tab character found in {file_path}")
                return None, None
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

def clean_text(text):
    """
    Clean text by removing punctuation and converting to lowercase.
    
    According to the requirements:
    - Replace all punctuation and numerals with space character
    - Convert all words to lowercase
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Create a translation table to replace punctuation and digits with spaces
    translator = str.maketrans(string.punctuation + string.digits, ' ' * (len(string.punctuation) + len(string.digits)))
    
    # Apply translation and convert to lowercase
    cleaned = text.translate(translator).lower()
    
    return cleaned

def tokenize_text(text):
    """
    Tokenize cleaned text into words.
    
    Args:
        text (str): Cleaned text
        
    Returns:
        list: List of words
    """
    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return words

def generate_bigrams(words):
    """
    Generate bigrams from a list of words.
    
    Args:
        words (list): List of words
        
    Returns:
        list: List of bigrams as tuples (word1, word2)
    """
    bigrams = []
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        bigrams.append(bigram)
    return bigrams

def filter_target_bigrams(bigrams, target_bigrams):
    """
    Filter bigrams to only include target bigrams.
    
    Args:
        bigrams (list): List of bigrams as tuples
        target_bigrams (set): Set of target bigrams as tuples
        
    Returns:
        list: List of matching target bigrams
    """
    return [bigram for bigram in bigrams if bigram in target_bigrams]

def bigram_to_string(bigram):
    """
    Convert a bigram tuple to string format.
    
    Args:
        bigram (tuple): (word1, word2)
        
    Returns:
        str: "word1 word2"
    """
    return f"{bigram[0]} {bigram[1]}"

def test_text_processing():
    """Test our text cleaning and tokenization."""
    print("\n=== Phase 2 Test: Text Processing ===")
    
    # Test with some sample text that has punctuation and numbers
    sample_text = "Hello, World! This is a test with 123 numbers and punctuation... How does it work?"
    
    print(f"Original text: {sample_text}")
    
    cleaned = clean_text(sample_text)
    print(f"Cleaned text: '{cleaned}'")
    
    words = tokenize_text(cleaned)
    print(f"Tokenized words: {words}")
    print(f"Word count: {len(words)}")
    
    # Test with actual file content (small sample)
    test_file = "data/devdata/5722018101.txt"
    if os.path.exists(test_file):
        doc_id, content = parse_document_file(test_file)
        if content:
            # Test with first 200 characters
            sample_content = content[:200]
            print(f"\nTesting with actual file content:")
            print(f"Original sample: {sample_content}")
            
            cleaned_sample = clean_text(sample_content)
            print(f"Cleaned sample: '{cleaned_sample}'")
            
            words_sample = tokenize_text(cleaned_sample)
            print(f"Sample words: {words_sample}")
            print(f"Sample word count: {len(words_sample)}")

def process_single_document(file_path):
    """
    Process a single document and create word counts.
    
    This is our basic "MapReduce" logic:
    - Map: document → (word, docID) pairs  
    - Reduce: group by word and count
    
    Args:
        file_path (str): Path to document file
        
    Returns:
        dict: {word: count} for this document, or None if failed
        str: document ID, or None if failed
    """
    doc_id, content = parse_document_file(file_path)
    
    if not doc_id or not content:
        return None, None
    
    # Clean and tokenize the content
    cleaned_content = clean_text(content)
    words = tokenize_text(cleaned_content)
    
    # Count word frequencies using Counter
    word_counts = Counter(words)
    
    return dict(word_counts), doc_id

def process_single_document_bigrams(file_path, target_bigrams):
    """
    Process a single document and create bigram counts for target bigrams only.
    
    Args:
        file_path (str): Path to document file
        target_bigrams (set): Set of target bigrams as tuples
        
    Returns:
        dict: {bigram_string: count} for this document, or None if failed
        str: document ID, or None if failed
    """
    doc_id, content = parse_document_file(file_path)
    
    if not doc_id or not content:
        return None, None
    
    # Clean and tokenize the content
    cleaned_content = clean_text(content)
    words = tokenize_text(cleaned_content)
    
    # Generate all bigrams
    all_bigrams = generate_bigrams(words)
    
    # Filter for target bigrams only
    target_bigrams_found = filter_target_bigrams(all_bigrams, target_bigrams)
    
    # Count bigram frequencies using Counter
    bigram_counts = Counter(target_bigrams_found)
    
    # Convert to string format for output
    bigram_string_counts = {}
    for bigram, count in bigram_counts.items():
        bigram_string = bigram_to_string(bigram)
        bigram_string_counts[bigram_string] = count
    
    return bigram_string_counts, doc_id

def create_inverted_index_entry(word_counts, doc_id):
    """
    Convert word counts for a single document into inverted index format.
    
    Args:
        word_counts (dict): {word: count} for a document
        doc_id (str): Document ID
        
    Returns:
        dict: {word: {doc_id: count}} - inverted index format
    """
    inverted_index = defaultdict(dict)
    
    for word, count in word_counts.items():
        inverted_index[word][doc_id] = count
    
    return dict(inverted_index)

def test_single_document_processing():
    """Test processing a single document completely."""
    print("\n=== Phase 3 Test: Single Document Processing ===")
    
    test_file = "data/devdata/5722018101.txt"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        return
    
    print(f"Processing: {test_file}")
    
    word_counts, doc_id = process_single_document(test_file)
    
    if word_counts and doc_id:
        print(f"✓ Successfully processed document!")
        print(f"Document ID: {doc_id}")
        print(f"Total unique words: {len(word_counts)}")
        print(f"Total word instances: {sum(word_counts.values())}")
        
        # Show some example word counts
        print("\nTop 10 most frequent words:")
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for word, count in top_words:
            print(f"  '{word}': {count}")
        
        # Test inverted index format
        inverted_index = create_inverted_index_entry(word_counts, doc_id)
        print(f"\nInverted index format test:")
        print(f"Example entries:")
        for i, (word, doc_dict) in enumerate(inverted_index.items()):
            if i < 3:  # Show first 3 words
                print(f"  '{word}': {doc_dict}")
            else:
                break
    else:
        print("✗ Failed to process document!")

def merge_inverted_indices(index1, index2):
    """
    Merge two inverted indices together.
    
    Args:
        index1 (dict): First inverted index {word: {docID: count, ...}, ...}
        index2 (dict): Second inverted index {word: {docID: count, ...}, ...}
        
    Returns:
        dict: Merged inverted index
    """
    merged = defaultdict(dict)
    
    # Add all entries from index1
    for word, doc_counts in index1.items():
        merged[word].update(doc_counts)
    
    # Add all entries from index2  
    for word, doc_counts in index2.items():
        merged[word].update(doc_counts)
    
    return dict(merged)

def process_directory(directory_path):
    """
    Process all .txt files in a directory and build complete inverted index.
    
    This simulates the full MapReduce job:
    - Map: Process each file independently
    - Reduce: Merge all results together
    
    Args:
        directory_path (str): Path to directory containing .txt files
        
    Returns:
        dict: Complete inverted index {word: {docID: count, docID: count, ...}, ...}
        list: List of processed document IDs
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found!")
        return {}, []
    
    inverted_index = {}
    processed_docs = []
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    txt_files.sort()  # Process in consistent order
    
    print(f"Found {len(txt_files)} .txt files in {directory_path}")
    
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {i+1}/{len(txt_files)}: {filename}...")
        
        # Process single document
        word_counts, doc_id = process_single_document(file_path)
        
        if word_counts and doc_id:
            # Convert to inverted index format
            doc_inverted_index = create_inverted_index_entry(word_counts, doc_id)
            
            # Merge with main index
            inverted_index = merge_inverted_indices(inverted_index, doc_inverted_index)
            processed_docs.append(doc_id)
            
            print(f"  ✓ Processed {doc_id}: {len(word_counts)} unique words, {sum(word_counts.values())} total words")
        else:
            print(f"  ✗ Failed to process {filename}")
    
    return inverted_index, processed_docs

def process_directory_bigrams(directory_path, target_bigrams):
    """
    Process all .txt files in a directory and build bigram inverted index.
    
    Args:
        directory_path (str): Path to directory containing .txt files
        target_bigrams (set): Set of target bigrams as tuples
        
    Returns:
        dict: Complete bigram inverted index {bigram: {docID: count, ...}, ...}
        list: List of processed document IDs
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found!")
        return {}, []
    
    inverted_index = {}
    processed_docs = []
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    txt_files.sort()  # Process in consistent order
    
    print(f"Found {len(txt_files)} .txt files in {directory_path}")
    
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {i+1}/{len(txt_files)}: {filename}...")
        
        # Process single document for bigrams
        bigram_counts, doc_id = process_single_document_bigrams(file_path, target_bigrams)
        
        if bigram_counts is not None and doc_id:
            # Convert to inverted index format
            doc_inverted_index = create_inverted_index_entry(bigram_counts, doc_id)
            
            # Merge with main index
            inverted_index = merge_inverted_indices(inverted_index, doc_inverted_index)
            processed_docs.append(doc_id)
            
            found_bigrams = len(bigram_counts)
            total_bigram_instances = sum(bigram_counts.values())
            print(f"  ✓ Processed {doc_id}: {found_bigrams} target bigrams found, {total_bigram_instances} total instances")
            
            # Show which bigrams were found
            if bigram_counts:
                for bigram, count in bigram_counts.items():
                    print(f"    '{bigram}': {count}")
        else:
            print(f"  ✗ Failed to process {filename}")
    
    return inverted_index, processed_docs

def test_multi_document_processing():
    """Test processing multiple documents from devdata directory."""
    print("\n=== Phase 4 Test: Multi-Document Processing ===")
    
    directory = "data/devdata"
    
    inverted_index, processed_docs = process_directory(directory)
    
    if inverted_index and processed_docs:
        print(f"\n✓ Successfully processed {len(processed_docs)} documents!")
        print(f"Document IDs: {sorted(processed_docs)}")
        print(f"Total unique words in index: {len(inverted_index)}")
        
        # Calculate total word instances across all documents
        total_instances = sum(sum(doc_counts.values()) for doc_counts in inverted_index.values())
        print(f"Total word instances across all documents: {total_instances}")
        
        # Show some example words that appear in multiple documents
        print(f"\nWords appearing in multiple documents:")
        multi_doc_words = [(word, doc_counts) for word, doc_counts in inverted_index.items() if len(doc_counts) > 1]
        multi_doc_words.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by number of documents
        
        for word, doc_counts in multi_doc_words[:5]:
            print(f"  '{word}': appears in {len(doc_counts)} documents - {doc_counts}")
        
        # Show top words by total frequency across all documents
        print(f"\nTop 10 words by total frequency across all documents:")
        word_totals = [(word, sum(doc_counts.values())) for word, doc_counts in inverted_index.items()]
        word_totals.sort(key=lambda x: x[1], reverse=True)
        
        for word, total_count in word_totals[:10]:
            doc_counts = inverted_index[word]
            print(f"  '{word}': {total_count} total ({len(doc_counts)} docs)")
            
    else:
        print("✗ Failed to process documents!")

def format_inverted_index_line(word, doc_counts):
    """
    Format a single line of the inverted index output.
    
    Format: word docID1:count1 docID2:count2 ...
    
    Args:
        word (str): The word
        doc_counts (dict): {docID: count, ...}
        
    Returns:
        str: Formatted line
    """
    # Sort document IDs for consistent output
    sorted_docs = sorted(doc_counts.items())
    
    # Create docID:count pairs
    doc_parts = [f"{doc_id}:{count}" for doc_id, count in sorted_docs]
    
    # Join everything together
    return f"{word} {' '.join(doc_parts)}"

def write_inverted_index(inverted_index, output_file):
    """
    Write inverted index to output file in required format.
    
    Args:
        inverted_index (dict): {word: {docID: count, ...}, ...}
        output_file (str): Path to output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Sort words alphabetically for consistent output
            sorted_words = sorted(inverted_index.keys())
            
            for word in sorted_words:
                doc_counts = inverted_index[word]
                line = format_inverted_index_line(word, doc_counts)
                f.write(line + '\n')
        
        return True
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        return False

def test_output_formatting():
    """Test output formatting with current devdata results."""
    print("\n=== Phase 6 Test: Output Formatting ===")
    
    directory = "data/devdata"
    inverted_index, processed_docs = process_directory(directory)
    
    if inverted_index:
        print(f"Creating sample output file...")
        
        # Create a sample with just a few words to see the format
        sample_words = ['computer', 'science', 'information', 'retrieval', 'the', 'and', 'university']
        sample_index = {}
        
        for word in sample_words:
            if word in inverted_index:
                sample_index[word] = inverted_index[word]
        
        # Show format examples in console
        print(f"\nSample output format:")
        for word in sorted(sample_index.keys()):
            doc_counts = sample_index[word]
            line = format_inverted_index_line(word, doc_counts)
            print(f"  {line}")
        
        # Write a small sample file
        sample_file = "sample_unigram_index.txt"
        if write_inverted_index(sample_index, sample_file):
            print(f"\n✓ Sample output written to: {sample_file}")
            
            # Show file stats
            with open(sample_file, 'r') as f:
                lines = f.readlines()
            print(f"Sample file contains {len(lines)} lines")
        else:
            print(f"✗ Failed to write sample file")
        
        # Also create a small complete file with first 100 words
        complete_sample_words = sorted(inverted_index.keys())[:100]
        complete_sample_index = {word: inverted_index[word] for word in complete_sample_words}
        
        complete_sample_file = "sample_unigram_index_100words.txt"
        if write_inverted_index(complete_sample_index, complete_sample_file):
            print(f"✓ Complete sample (100 words) written to: {complete_sample_file}")
        
        # Show some statistics about what the full file would look like
        print(f"\nFull devdata unigram index statistics:")
        print(f"  Total words: {len(inverted_index)}")
        print(f"  Total documents: {len(processed_docs)}")
        
        # Show distribution of how many documents each word appears in
        doc_counts_distribution = {}
        for word, doc_counts in inverted_index.items():
            num_docs = len(doc_counts)
            doc_counts_distribution[num_docs] = doc_counts_distribution.get(num_docs, 0) + 1
        
        print(f"  Word distribution across documents:")
        for num_docs in sorted(doc_counts_distribution.keys()):
            count = doc_counts_distribution[num_docs]
            print(f"    {count} words appear in {num_docs} document(s)")
            
    else:
        print("✗ No inverted index available!")

def test_bigram_processing():
    """Test bigram processing with target bigrams."""
    print("\n=== Phase 5 Test: Bigram Processing ===")
    
    # Define our 5 target bigrams
    target_bigram_strings = [
        "computer science",
        "information retrieval", 
        "power politics",
        "los angeles",
        "bruce willis"
    ]
    
    # Convert to tuples for processing
    target_bigrams = set()
    for bigram_str in target_bigram_strings:
        word1, word2 = bigram_str.split()
        target_bigrams.add((word1, word2))
    
    print(f"Target bigrams: {target_bigram_strings}")
    
    # Test bigram generation with sample text
    sample_text = "Computer science and information retrieval are important. Power politics affects everyone."
    print(f"\nTesting with sample text: {sample_text}")
    
    cleaned = clean_text(sample_text)
    words = tokenize_text(cleaned)
    print(f"Cleaned words: {words}")
    
    all_bigrams = generate_bigrams(words)
    print(f"All bigrams: {[bigram_to_string(bg) for bg in all_bigrams]}")
    
    found_target_bigrams = filter_target_bigrams(all_bigrams, target_bigrams)
    print(f"Target bigrams found: {[bigram_to_string(bg) for bg in found_target_bigrams]}")
    
    # Test with actual devdata directory
    directory = "data/devdata"
    bigram_index, processed_docs = process_directory_bigrams(directory, target_bigrams)
    
    if bigram_index:
        print(f"\n✓ Bigram processing completed!")
        print(f"Documents processed: {processed_docs}")
        print(f"Target bigrams found in corpus: {len(bigram_index)}")
        
        # Show results for each target bigram
        print(f"\nTarget bigram results:")
        for bigram_str in target_bigram_strings:
            if bigram_str in bigram_index:
                doc_counts = bigram_index[bigram_str]
                total_count = sum(doc_counts.values())
                print(f"  '{bigram_str}': {total_count} total occurrences in {len(doc_counts)} documents")
                print(f"    Details: {doc_counts}")
            else:
                print(f"  '{bigram_str}': NOT FOUND")
        
        # Create sample bigram output file
        if bigram_index:
            sample_bigram_file = "selected_bigram_index.txt"
            if write_inverted_index(bigram_index, sample_bigram_file):
                print(f"\n✓ Sample bigram output written to: {sample_bigram_file}")
            else:
                print(f"✗ Failed to write sample bigram file")
    else:
        print("✗ No bigram index created!")

def test_file_parsing():
    """Test our file parsing with one of the dev data files."""
    test_file = "data/devdata/5722018101.txt"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        return
    
    print("=== Phase 1 Test: File Parsing ===")
    print(f"Testing file: {test_file}")
    
    doc_id, content = parse_document_file(test_file)
    
    if doc_id and content:
        print(f"✓ Successfully parsed file!")
        print(f"Document ID: {doc_id}")
        print(f"Content length: {len(content)} characters")
        print(f"Content preview (first 100 chars): {content[:100]}...")
        print(f"Content preview (last 100 chars): ...{content[-100:]}")
    else:
        print("✗ Failed to parse file!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            print("Running test functions...")
            test_file_parsing()
            test_text_processing()
            test_single_document_processing()
            test_multi_document_processing()
            test_output_formatting()
            test_bigram_processing()
            exit()
        elif sys.argv[1] == "dev" or sys.argv[1] == "devdata":
            # Use devdata (5 files) for quick testing
            data_dir = 'data/devdata'
            output_suffix = '_dev'
            print("=" * 60)
            print("INVERTED INDEX GENERATION - DEVDATA (TESTING)")
            print("=" * 60)
            print("Using devdata (5 files) for quick testing...")
        elif sys.argv[1] == "full" or sys.argv[1] == "fulldata":
            # Use fulldata (74 files) for final submission
            data_dir = 'data/fulldata'
            output_suffix = ''
            print("=" * 60)
            print("INVERTED INDEX GENERATION - FULLDATA (FINAL)")
            print("=" * 60)
            print("Using fulldata (74 files) for final submission...")
        else:
            print("Usage:")
            print("  python3 main.py              # Default: use fulldata (74 files)")
            print("  python3 main.py dev          # Use devdata (5 files) for testing")
            print("  python3 main.py full         # Use fulldata (74 files) explicitly")
            print("  python3 main.py test         # Run test functions only")
            exit()
    else:
        # Default: use fulldata for final submission
        data_dir = 'data/fulldata'
        output_suffix = ''
        print("=" * 60)
        print("INVERTED INDEX GENERATION - FULLDATA (DEFAULT)")
        print("=" * 60)
        print("Using fulldata (74 files) by default...")
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 1. Generate unigram index
    print(f"\n1. Generating unigram index from {data_dir}...")
    try:
        unigram_index, processed_docs = process_directory(data_dir)
        unigram_filename = f'unigram_index{output_suffix}.txt'
        unigram_output_path = os.path.join(output_dir, unigram_filename)
        write_inverted_index(unigram_index, unigram_output_path)
        print(f"   ✓ Generated {unigram_output_path}")
        print(f"   ✓ Processed {len(processed_docs)} documents")
        print(f"   ✓ Created index with {len(unigram_index)} unique words")
    except Exception as e:
        print(f"   ✗ Error generating unigram index: {e}")
    
    # 2. Generate bigram index
    print(f"\n2. Generating bigram index from {data_dir}...")
    try:
        target_bigram_strings = [
            'computer science',
            'information retrieval', 
            'power politics',
            'los angeles',
            'bruce willis'
        ]
        
        target_bigrams = set()
        for bigram_str in target_bigram_strings:
            word1, word2 = bigram_str.split()
            target_bigrams.add((word1, word2))
        
        bigram_index, processed_docs = process_directory_bigrams(data_dir, target_bigrams)
        bigram_filename = f'selected_bigram_index{output_suffix}.txt'
        bigram_output_path = os.path.join(output_dir, bigram_filename)
        write_inverted_index(bigram_index, bigram_output_path)
        print(f"   ✓ Generated {bigram_output_path}")
        print(f"   ✓ Processed {len(processed_docs)} documents")
        print(f"   ✓ Found {len(bigram_index)} target bigrams")
        
        # Show bigram counts
        print("   ✓ Bigram counts:")
        for bigram in sorted(bigram_index.keys()):
            total_count = sum(bigram_index[bigram].values())
            print(f"      '{bigram}': {total_count} occurrences")
            
    except Exception as e:
        print(f"   ✗ Error generating bigram index: {e}")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    if output_suffix == '_dev':
        print(f"\nDevelopment test files generated in '{output_dir}' folder:")
        print(f"  📄 {output_dir}/unigram_index_dev.txt - Test unigram index")
        print(f"  📄 {output_dir}/selected_bigram_index_dev.txt - Test bigram index")
        print("\n🚀 Ready for final run? Use: python3 main.py full")
    else:
        print(f"\nSubmission files generated in '{output_dir}' folder:")
        print(f"  📄 {output_dir}/unigram_index.txt - Unigram inverted index")
        print(f"  📄 {output_dir}/selected_bigram_index.txt - Bigram inverted index") 
        print("  📄 main.py - Source code")
        print(f"\n📸 IMPORTANT: Take screenshots of the '{output_dir}' folder for homework submission!")
    
    print("\nUsage options:")
    print("  python3 main.py dev   # Quick test with devdata (5 files)")
    print("  python3 main.py full  # Final run with fulldata (74 files)")
    print("  python3 main.py test  # Run test functions only")
