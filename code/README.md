## Constructing GQA-OOD dataset

We provide the code necessary for constructing the GQA-OOD dataset. 

### 1) Download [GQA's original questions](https://cs.stanford.edu/people/dorarad/gqa/download.html)

Then, place them in gqa/. In the paper, we use the version 1.2 of the dataset.

### 2) Construct the GQA-OOD dataset

To extract questions used for evaluating with the GQA-OOD benchmark, launch:

> python build_gqa_ood.py --alpha 0.2 --split val

This command will contruct the three GQA-OOD sets, namely 'all', 'head' and 'tail', for the validation split.
You will find them in ../data/.

## Evaluating on GQA-OOD

We also provide some scripts to evaluate your model's predictions on GQA-OOD. The predictions must respect the standard GQA format.

### Measuring acc-all, acc-tail and acc-head

To obtain the three metrics described in the paper, launch:

> python evaluation.py --ood_test --predictions [prediction path (on ood_testdev_all or gqa_testdev)]

Those are measured on the testdev split of GQA-OOD.

### Drawing acc vs. alpha plot

To draw analyse the influence of alpha against the model accuracy, launch:

> python evaluation.py --eval_tail_size --predictions [prediction path (on ood_val_all or gqa_val)]

This analyse is performed on the validation split of GQA-OOD. A plot will be saved in plot/.

#### Please, find the python requirements in requirement.txt

## License

All provided source codes are the property of the authors (copyright 2020), and are licensed under the GNU General Public License, version 3 (GPL-3.0). By downloading them, you agree to use them only for non-commercial purposes and to abide by all terms of the GNU GPL-3.0 License.
More particularly, by requesting access to the sources, you commit:
- to report to the authors any content which you consider inappropriate or unsuitable (including without limitation offensive, harmful, threatening, violent or otherwise objectionable content)
- to remove any such content
- to refrain from making claims against the authors whatever the basis.
