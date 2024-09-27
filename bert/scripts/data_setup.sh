#!/bin/bash

# Download the Cornell Movie Dialogs Corpus
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

# Unzip the downloaded file quietly
unzip -qq cornell_movie_dialogs_corpus.zip

# Remove the zip file
rm cornell_movie_dialogs_corpus.zip

# Create a datasets directory if it doesn't exist
mkdir -p datasets

# Move the relevant files to the datasets directory
mv cornell\ movie-dialogs\ corpus/movie_conversations.txt ./datasets
mv cornell\ movie-dialogs\ corpus/movie_lines.txt ./datasets
