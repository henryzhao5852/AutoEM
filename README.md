
# [Auto-EM](https://dl.acm.org/doi/10.1145/3308558.3313578) source code
This repo hosts the source code used in the paper "[Auto-EM: End-to-end Fuzzy Entity-Matching using Pre-trained Deep Models and Transfer Learning](https://dl.acm.org/doi/10.1145/3308558.3313578)", published in the Web conference (WWW) 2019.

## What are included

- Source code for entity typing and entity matching models


## Dependency Installation
Run `pip install -r requirements.txt`.


## Data files
We leverage the entity-type and entity-synonym data extracted from Microsoft's properietary KB, to train Auto-EM. While we could not directly release the data due to the proprietary nature of the KB, similar data are readily available in open-source KBs. For example, in Wikidata, each entity has an attribute called `also known as`, which lists alias/synonyms for the given entity (e.g., the [Wikidata page](https://www.wikidata.org/wiki/Q5284) for 'Bill Gates' lists 'William Gates', 'William Henry Gates III' as synonyms. Similarly, the [Freebase](https://developers.google.com/freebase) also has an `alias` attribute for entity synonyms that can be used to train Auto-EM models). 

## Data format
Although we could not release our training data, we provide sample files (under `data/`) to demonstrate the data format accepted by the code (so that data extracted from Wikidata or Freebase can be made into the same format and run).

For **entity typing** (given an entity name, predict its entity types -- e.g., 'Bill Gates' is of type 'person', 'politician', etc.)  
- The format for each line in `data/sample_data_type.txt` is: `entity_name	lb1___lb2___lb3`   
- Here, entity_name and its corresponding entity_types are seperated by a tab character, with multiple type-ids are seperated by underscores.

For **entity matching** (given two entity names, predict whether they are a match or not -- e.g., 'Bill Gates' and 'William Gates' is a match, but 'Bill Gates' and 'Bill Gates Sr.' is not a match)  
- The format for each line in `data/sample_data_match.txt` is: `entity_name	pos_alias	neg_alias1___neg_alias2___neg_alias3`  
- Here, each line has three components, separated by tabs. The first component is one canonical entity_name. The second component is its positive alias/synonyms. The third component is its negative alias/synonyms. 



## Entity Typing Model Training and Testing
For entity typing, `cd type` and run `python alias_type_train.py` to train a entity typing model, with following input arguments

- `--train-file` for training data
- `--dev-file` for dev data
- `--test-file` for test data
- `--save-model` for saved model path
- `--load-model` for loading model checkpoint from the path

Once the model is trained, run `python alias_type_train.py --test` (with your saved model) to make predictions on the test set.

Also see `type/alias_type_train.py` for command-line arguments to specify training file, dev file, test file, model saving location, etc.

## Entity Matching Model Training and Testing


For entity matching, `cd matching` and run `python hybrid_train.py` to train a entity matching model, with following input arguments

- `--train-file` for training data
- `--dev-file` for dev data
- `--test-file` for test data
- `--save-model` for saved model path
- `--load-model` for loading model checkpoint from the path

Once the model is trained, run `python hybrid_train.py --test` (with your saved model) to make predictions on the test set.

Also see `matching/hybrid_train.py` for command-line arguments to specify training file, dev file, test file, model saving location, etc.


## Trademarks 
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Contact
If you have questions, suggestions and bug reports, please email the authors in the [paper](https://dl.acm.org/doi/10.1145/3308558.3313578).

