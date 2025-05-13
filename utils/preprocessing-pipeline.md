# Text Processing Pipeline Documentation

## Overview

The `TextProcessor` class in text_processing.py provides a comprehensive text preprocessing pipeline for natural language processing tasks. This class is designed to clean and normalize text data through multiple steps.

## Class Structure

### Initialization

```python
processor = TextProcessor(dataset, column_name)
```

- `dataset`: pandas DataFrame containing the text data
- `column`: name of the column containing text to process

### Main Methods

- `transform()`: Processes entire column of the dataset
- `process(text)`: Processes a single text string
- `process_text(text)`: Core processing pipeline

## Processing Pipeline Steps

1. **Case Normalization** (`to_lower`)

   ```python
   text = "Hello World" -> "hello world"
   ```

2. **Contraction Expansion** (`expand_contractions`)

   ```python
   text = "don't" -> "do not"
   ```

3. **Digit Removal** (`remove_digits`)

   ```python
   text = "hello123" -> "hello"
   ```

4. **Unicode Normalization** (`remove_unicode`)
   - Converts Unicode characters to ASCII
   - Handles diacritics and special characters

5. **Non-ASCII Punctuation Removal** (`remove_non_ascii_punctuation`)
   - Removes characters outside ASCII range
   - Pattern used: `[^\x00-\x7F]`

6. **Special Character Cleaning** (`remove_special_characters`)
   - Keeps: letters, numbers, spaces, hyphens, apostrophes
   - Pattern used: `[^a-zA-Z0-9\s\'-]`

7. **Whitespace Normalization** (`remove_extra_spaces`)
   - Removes redundant spaces
   - Trims leading/trailing whitespace

8. **Token Addition**
   - Adds `<start>` and `<end>` tokens
