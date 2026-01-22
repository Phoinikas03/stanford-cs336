# TinyStories train
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

# # TinyStories valid
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt


# owt-sample train
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz

# owt-sample valid
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

mkdir -p ../dataset/tinystories
mkdir -p ../dataset/openwebtext

mv TinyStoriesV2-GPT4-train.txt ../dataset/tinystories/TinyStoriesV2-GPT4-train.txt
mv TinyStoriesV2-GPT4-valid.txt ../dataset/tinystories/TinyStoriesV2-GPT4-valid.txt
mv owt_train.txt ../dataset/openwebtext/owt_train.txt
mv owt_valid.txt ../dataset/openwebtext/owt_valid.txt