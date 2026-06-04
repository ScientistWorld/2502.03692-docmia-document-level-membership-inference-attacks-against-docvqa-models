# Experiment Overview

## scaled_whitebox_docmia

Tests whether document-level membership can be inferred from a trained
DocVQA-style model on a balanced set of member and non-member documents. This
experiment is the minimum setting for the paper's central privacy claim: the
system should identify training documents while using only the target documents
and their associated question-answer pairs, not an auxiliary shadow dataset.
The expanded version also compares multiple white-box optimization signals
under the same query budget and target-model checkpoint.
