![JIGEN](https://github.com/fmcarlucci/JigenDG/blob/master/jigsaw2-1.png)


## Setup Instructions

```
git clone https://github.com/vdevmcitylp/JigenDG.git
cd JigenDG
```

### Environment Setup

Assumes Python 3.6 is installed at /usr/bin/python3.6 
To check, `find /usr/bin/python3.6` 

virtualenv -p /usr/bin/python3.6 pyji36

To activate, `. pyji36/bin/activate`

`pip install -r requirements.txt`

### Generate Stylized PACS Dataset

Follow instructions in [this](https://github.com/vdevmcitylp/cross-domain-self-supervision) repository to generate StylizedPACS dataset.

### Generate Stylized txt_lists

Assumes that data/txt_lists are the ones from the author's repository.

chmod +x stylized_pacs_setup.sh
./stylized_pacs_setup.sh --dataset_root <Full path to folder where PACS and StylizedPACS are located>

Check training_recipes.md for commands