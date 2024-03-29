#!/usr/bin/env python

from pathlib import Path
import json
import csv
from typing import Dict, List, Union, Tuple, Any
import random
from datetime import datetime
import argparse


from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

__author__ = 'Samuel Lipping -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['evaluate_metrics']


def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file

    :param data: Data to write
    :type data: list[dict[str, any]] or dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    with path.open("w") as f:
        json.dump(data, f)


def reformat_to_coco(predictions: List[str],
                     ground_truths: List[List[str]],
                     ids: Union[List[int], None] = None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format

    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer is used
    :type ids: list[int] or None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = range(len(predictions))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    for audio_id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': p
        })

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref


def evaluate_metrics_from_files(pred_file: Union[Path, str],
                                ref_file: Union[Path, str]) \
        -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """ Evaluate the translation metrics from annotation files with the coco lib
    Follows the example in the repo.

    :param pred_file: File with predicted captions
    :type pred_file: Path or str
    :param ref_file: File with reference captions
    :type ref_file: Path or str
    :return: Tuple with metrics for the whole dataset and per-file metrics
    :rtype: tuple[dict[str, float], dict[int, dict[str, float]]]
    """
    # Load annotations from files
    coco = COCO(str(ref_file))
    cocoRes = coco.loadRes(str(pred_file))

    # Create evaluation object and evaluate metrics
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['audio_id'] = cocoRes.getAudioIds()
    cocoEval.evaluate(verbose=False)

    # Make dict from metrics
    metrics = dict(
        (m, s) for m, s in cocoEval.eval.items()
    )
    return metrics, cocoEval.audioToEval


def evaluate_metrics_from_lists(predictions: List[str],
                                ground_truths: List[List[str]],
                                ids: Union[List[int], None] = None) \
        -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Evaluate metrics from lists of predictions and ground truths

    :param predictions: List of prediction captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions (one five-caption list per file)
    :type ground_truths: list[list[str]]
    :param ids: Ids for the audio files. If not given, a running integer is used
    :type ids: list[int] or None
    :return: Tuple with metrics for the whole dataset and per-file metrics
    :rtype: tuple[dict[str, float], dict[int, dict[str, float]]]
    """
    assert(len(predictions) == len(ground_truths))
    assert(all([len(i) == 5 for i in ground_truths]))

    # Running int for id if not given
    if ids is None:
        ids = range(len(predictions))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred, ref = reformat_to_coco(predictions, ground_truths, ids)

    # Write temporary files for the metric evaluation
    tmp_dir = Path('tmp')

    if not tmp_dir.is_dir():
        tmp_dir.mkdir()

    unique_id = f'{random.randint(0, 1e6)}_{datetime.now()}'

    ref_file = tmp_dir.joinpath(f'{unique_id}_ref.json')
    pred_file = tmp_dir.joinpath(f'{unique_id}_pred.json')

    write_json(ref, ref_file)
    write_json(pred, pred_file)

    metrics, per_file_metrics = evaluate_metrics_from_files(pred_file, ref_file)

    # Delete temporary files
    ref_file.unlink()
    pred_file.unlink()

    return metrics, per_file_metrics


def check_and_read_csv(path: Union[str, Path, List[Dict[str, str]]]) \
        -> List[Dict[str, str]]:
    """ If input is a file path, returns the data as a list of dicts (as returned by DictReader)
    Otherwise just returns the input

    :param path: Input file or its contents (as given by DictReader)
    :type path: Path, str or list[dict[str, str]]
    :return: File contents
    :rtype: list[dict[str, str]]
    """
    if not isinstance(path, list):
        if isinstance(path, str):
            path = Path(path)

        with path.open('r') as f:
            reader = csv.DictReader(f, dialect='unix')

            result = [row for row in reader]
    else:
        result = path

    return result


def combine_single_and_per_file_metrics(single_metrics: Dict[str, float],
                                        per_file_metrics: Dict[int, Dict[str, float]],
                                        file_names: List[str]) \
        -> Dict[str, Dict[str, Any]]:
    """ Reformat single (one for whole dataset) and per-file metrics into
    {
      <metric_name>:{
          'score': <single metric value>,
          'scores': {
              <file_name>: <per-file metric value>
          }
      }
    }

    :param single_metrics: Evaluated single metrics
    :type single_metrics: dict[str, float]
    :param per_file_metrics: Evaluated per-file metrics
    :type per_file_metrics: dict[int, dict[str, float]]
    :param file_names: List of file names in the order they were given to the metric evaluator
    :type file_names: list[str]
    :return: Evaluated metrics in one data structure
    :rtype: dict[str, dict[str, any]]
    """
    total_metrics = {}
    for metric, score in single_metrics.items():
        total_metrics[metric] = {
            'score': score,
            'scores': {}
        }
    for file_idx, metric_dict in per_file_metrics.items():
        file_name = file_names[file_idx]
        for metric in total_metrics.keys():
            if metric == 'SPICE':
                value = metric_dict[metric]['All']['f']
            else:
                value = metric_dict[metric]
            total_metrics[metric]['scores'][file_name] = value

    return total_metrics


def evaluate_metrics(prediction_file: Union[str, Path, List[Dict[str, str]]],
                     reference_file: Union[str, Path, List[Dict[str, str]]],
                     nb_reference_captions: int = 5) \
        -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """ Evaluates metrics from the predictions and reference captions.

    Evaluates BLEU1-4, CIDEr, METEOR, ROUGE_L, SPICE, and SPIDEr using
    code from https://github.com/tylin/coco-caption

    :param prediction_file: Input file (or file contents, as given by DictReader) \
                            with predicted captions
    :type prediction_file: Path | str | list[dict[str, str]]
    :param reference_file: Input file (or file contents, as given by DictReader) \
                           with reference captions
    :type reference_file: Path | str | list[dict[str, str]]
    :param nb_reference_captions: Number of reference captions
    :type nb_reference_captions: int
    :return: A dict with keys the names of the metrics. Each metric\
             has as value a dict, with keys `score` and `scores`. The\
             `score` key, has as a value the score of the corresponding\
             metric, for the whole set of files. The `scores` keys, has\
             as a value, a dict with keys the file names of the files, and\
             values the value of the score for the corresponding file.
    :rtype: dict[str, dict[str, float|dict[str, float]]
    """
    prediction_file = check_and_read_csv(prediction_file)  # 什么都不做，还是返回结果的list
    reference_file = check_and_read_csv(reference_file)    # 什么都不做，还是返回结果的list

    prediction_file.sort(key=lambda the_row: the_row['file_name'])
    reference_file.sort(key=lambda the_row: the_row['file_name'])

    # Make reference file contents indexable by file name
    reference_dict = {}
    for row in reference_file:
        reference_dict[row['file_name']] = row

    # Make sure that all the files in the prediction file exist also in the reference file
    file_names = [row['file_name'] for row in prediction_file]
    assert(
        all(
            file_name in reference_dict for file_name in file_names
        )
    )

    predictions = []
    ground_truths = []
    for row in prediction_file:
        file_name = row['file_name']
        predictions.append(row['caption_predicted'])

        cap_names = ['caption_{:1d}'.format(i) for i in range(1, nb_reference_captions+1)]

        ground_truths.append([reference_dict[file_name][cap] for cap in cap_names])

    metrics, per_file_metrics = evaluate_metrics_from_lists(predictions, ground_truths)

    total_metrics = combine_single_and_per_file_metrics(
        metrics, per_file_metrics, file_names
    )

    return {
        key.lower(): value for key, value in total_metrics.items()
    }
    

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics for predictions.')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to the file containing predictions.')
    parser.add_argument('--gt_file', type=str, required=True, help='Path to the file containing ground truth captions.')
    return parser.parse_args()


if __name__ == '__main__': 

    args = parse_args()

    pred_file = args.pred_file
    gt_file = args.gt_file

    pred_captions = []
    gt_captions = []

    with open(pred_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            audio_id, caption = line.split('\t')
            pred_captions.append(caption)
 
    captions_per_audio = []
    with open(gt_file, 'r', encoding='utf-8') as file: 
        for idx, line in enumerate(file): 
            if idx % 5 == 0 and idx != 0: 
                gt_captions.append(captions_per_audio)
                captions_per_audio = []

            line = line.strip()
            audio_id, caption = line.split('\t')
            captions_per_audio.append(caption)


        if captions_per_audio is not []: 
            gt_captions.append(captions_per_audio)

    print('--evaluation start--')
    metrics, per_file_metrics = evaluate_metrics_from_lists(pred_captions, gt_captions)

    print("Results shown as below: ")
    print(metrics)

    # spider = metrics['spider']['score']
    # cider = metrics['cider']['score']
    # print(f'spider: {spider}, cider: {cider}')
