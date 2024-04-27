# An unofficial reproduction of VALL-E-X
We refer to the repository of https://github.com/Plachtaa/VALL-E-X to open an unofficial reproduction of VALLEX.

## Checkpoints
Pretrained model can be found at [Google driven](https://drive.google.com/drive/folders/1wCTffPnSsiHpthaX-yzUne1dTBAMI_JV?usp=drive_link).


## Decode with checkpoints
```shell
# first modify the model_home in following scrip to the location of downloaded/pretrained models.
# second diy the prompt_txt, prompt_audio, target_txt with a corresponding language id
bash examples\\vallex\\scripts\\inference.sh
```

## Data preparation
Vallex is trained on the dataset containing discrete speech tokens and text tokens.


* Prepare a "info.tsv" file as following (file \t duration), containing speech path and duration of each speech.
    ```txt
    SPEECH_PATH1    DURATION1
    SPEECH_PATH2    DURATION2
    SPEECH_PATH3    DURATION3
    ......
    ```
* Extract Codec according to the "info.tsv" 
    ```shell
    bash examples/vallex/data_pretreatment/extract_codec.sh
    ```
    We can obtain 8 "codec[i].tsv" files, 0~7 (i)-th layer of codecs are separately saved into "codec[i].tsv"
    ```txt
    304 123 453 255 256 345 124 666 543 ...
    654 662 543 463 674 537 273 473 973 ...
    355 345 766 255 234 768 275 785 102 ...
    ......
    ```

* Prepare the text ("trans.tsv") file with each line corresponding to the speech

    ```txt
    Text for SPEECH1
    Text for SPEECH2
    Text for SPEECH3
    ......
    ```
    Next, we need convert the text into tokens via tools like BPE/G2P/..., and it's saved as "st.tsv"
    ```txt
    1521 467 885 2367 242 ...
    2362 3261 356 167 1246 2364 ...
    1246 123 432 134 53 13 ...
    ......
    ```

* Convert data (codec[i].tsv and st.tsv) into binary file for fast reading

    ```shell
    # We use the fairseq tool to achieve this convertion process
    python /home/wangtianrui/codes/fairseq/fairseq_cli/preprocess.py \
        --only-source \
        --trainpref /home/wangtianrui/develop_dataset/st.tsv \
        --destdir /home/wangtianrui/develop_dataset/data_bin  \
        --thresholdsrc 0 \
        --srcdict /home/wangtianrui/develop_dataset/dict.st.txt \
        --workers `cat /proc/cpuinfo| grep "processor"| wc -l`

    for ((i=0;i<=7;i++))
    do
    echo $i
    outname=train.at${i}.zh
    python /home/wangtianrui/codes/fairseq/fairseq_cli/preprocess.py \
    --only-source \
    --trainpref codec${i}.tsv \
    --destdir $outdir \
    --thresholdsrc 0 \
    --srcdict /home/wangtianrui/develop_dataset/dict.at.txt \
    --workers `cat /proc/cpuinfo| grep "processor"| wc -l`
    done
    ```
    where dict.at.txt and dict.st.txt are simple idx-to-idx rows of speech discrete tokens and text tokens, as shown in examples/vallex/data_pretreatment 

In this way, we can train the vallex with the dataset_config.train_data_path set as the home_path of binary files. We also release a tiny dataset for reference at [Google driven](https://drive.google.com/drive/folders/1wCTffPnSsiHpthaX-yzUne1dTBAMI_JV?usp=drive_link).

## Train a new AR model

After pretreated dataset, modify the "train_data_path" in following script, you can start for your training or finetuning.

```shell
bash examples\\vallex\\scripts\\vallex.sh
```
