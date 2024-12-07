import os
from collections import Counter

# 音素の頻度分析用コード
def analyze_phonemes_from_lab(file_list, directory_path):
    """
    Analyze phonemes from specific .lab files listed in file_list.
    The file names in file_list are prefixed with 'LFROI_', and the corresponding
    .lab files in the directory are suffixed versions.
    """
    # Store all phonemes
    all_phonemes = []
    
    # Process each file listed in file_list
    for file_name in file_list:
        # Extract the relevant portion for the .lab file
        lab_file = file_name.replace('LFROI_', '') + '.lab'
        file_path = os.path.join(directory_path, lab_file)
        
        if not os.path.exists(file_path):
            print(f"Warning: {lab_file} does not exist in {directory_path}. Skipping.")
            continue
        
        # Read the .lab file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Split the line into columns (assuming tab or space-separated)
                parts = line.strip().split()
                if len(parts) < 3:  # Skip invalid lines
                    continue
                
                # The phoneme is typically the last column
                phoneme = parts[-1]
                all_phonemes.append(phoneme)
    
    # Get unique phonemes
    unique_phonemes = sorted(set(all_phonemes))
    
    # Count frequencies
    phoneme_counts = Counter(all_phonemes)
    
    # Sort by frequency (most common first)
    sorted_counts = sorted(phoneme_counts.items(), key=lambda x: (-x[1], x[0]))
    
    return {
        'unique_phonemes': unique_phonemes,
        'total_phonemes': len(all_phonemes),
        'unique_count': len(unique_phonemes),
        'frequency': sorted_counts
    }

def load_file_list(file_path):
    """
    Load the list of filenames (with the 'LFROI_' prefix) from the given file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def print_analysis(analysis):
    """Print analysis results in a formatted way"""
    print(f"Total phonemes found: {analysis['total_phonemes']}")
    print(f"Unique phonemes count: {analysis['unique_count']}")
    
    print("\nUnique phonemes:")
    print(", ".join(analysis['unique_phonemes']))
    
    print("\nPhoneme frequencies (sorted by count):")
    print("{:<10} {:<10} {:<10}".format("Phoneme", "Count", "Percentage"))
    print("-" * 30)
    for phoneme, count in analysis['frequency']:
        percentage = (count / analysis['total_phonemes']) * 100
        print("{:<10} {:<10} {:.2f}%".format(phoneme, count, percentage))

def save_results(analysis, output_file):
    """Save analysis results to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total phonemes: {analysis['total_phonemes']}\n")
        f.write(f"Unique phonemes: {analysis['unique_count']}\n\n")
        
        f.write("All unique phonemes:\n")
        f.write(", ".join(analysis['unique_phonemes']) + "\n\n")
        
        f.write("Frequency distribution:\n")
        f.write("{:<10} {:<10} {:<10}\n".format("Phoneme", "Count", "Percentage"))
        f.write("-" * 30 + "\n")
        for phoneme, count in analysis['frequency']:
            percentage = (count / analysis['total_phonemes']) * 100
            f.write("{:<10} {:<10} {:.2f}%\n".format(phoneme, count, percentage))

if __name__ == "__main__":
    # File containing the list of names prefixed with 'LFROI_'
    file_list_path = "unseen_train.txt"
    directory_path = "processed"  # Directory containing .lab files
    output_file = "phoneme_analysis.txt"
    
    # Load the file list from unseen_train.txt
    file_list = load_file_list(file_list_path)
    
    # Run analysis on the specified files
    results = analyze_phonemes_from_lab(file_list, directory_path)
    
    # Print results to console
    print_analysis(results)
    
    # Save results to file
    save_results(results, output_file)
