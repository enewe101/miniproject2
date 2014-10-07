# Mini Project #2
---

Second mini project for COMP 598

## Datasets description

processed\_train\_input.csv and processed\_test\_input.csv contain the processed input for the training and test examples, respectively. The processing steps include the following (in order):
* Remove erroneous examples in the training set (There are three examples that are misclassified as simply 'category').
* Separate LaTeX commands from their string arguments, treating the commands as any other word.
* Remove punctuation from the examples.
* Convert all words to lowercase.
* Lemmatize and stem words (The WordNet Lemmatizer and the Porter algorithm for stemming were used).
* Omit stop words from the examples (e.g. 'the', 'and', etc.)

The train\_input\_wf.txt and test\_input\_wf.txt files contain the word frequency counts for the (processed) training and test input sets, respectively. The format of these files is JSON, with each line corresponding to one example, with a number of entries separated by commas. Each entry has the form 'word:n', where 'word' corresponds to a word that appears in the example and 'n' the frequency of occurrence. This allows only the words with non-zero occurrence to be listed for each example. The very first entry of each line corresponds to the example id. Here's an example line: '{"id":1,"paper":1,"version":1,"thi":1,"ha":2,'new":1,"publish":1,"withdrawn":1}'. '1' is the id of the example and the entry '"thi":1' signifies the word 'thi' appears in the example once. One remark: Numbers and words with single occurrence in the entire corpous, were excluded from consideration in the word counting. I know this format is awkward, but unlike the .csv format, it avoids including a buttload of 0's (The vectors in the .csv format would be mostly filled with zero-entries).
