# Evaluates an AMP tempo submission on the challenge server,
# computing the mean p-score over all keys present in
# the reference. Refer to the 'mir_eval' library, if you
# have trouble producing predictions in the correct format.
import argparse
import json
import os
import numpy as np
import mir_eval


def prepare_truth(tempi):
    """
    Prepare ground truth tempo data for evaluation.
    Input can be:
    - [tempo] (single tempo, all annotators agreed)
    - [tempo1, tempo2, weight] (two tempi with weight for slower tempo)
    """
    if len(tempi) == 3:
        # Two tempi with weight: [slower_tempo, faster_tempo, weight_for_slower]
        tempi, weight = tempi[:2], tempi[2]
    elif len(tempi) == 1:
        # Single tempo: create second tempo as double, weight = 1.0 (all agreed)
        tempi, weight = [tempi[0], tempi[0] * 2], 1.0
    else:
        # Assume two tempi with equal weight if no weight provided
        tempi, weight = tempi[:2], 0.5
    return np.asarray(tempi), weight


def prepare_preds(tempi):
    """
    Prepare predicted tempo data for evaluation.
    If only one tempo provided, create second as double.
    """
    if len(tempi) < 2:
        tempi = [tempi[0], tempi[0] * 2]
    return np.asarray(tempi[:2])  # Only use first two predictions


def evaluate_loop(submission, target):
    """
    Evaluate tempo predictions using mir_eval.tempo.detection
    with ±8% tolerance as specified in the challenge rules.
    """
    sum_p = 0.
    for target_key, target_value in target.items():
        if target_key in submission:
            reference_tempi = target_value['tempo']
            estimated_tempi = submission[target_key]['tempo']

            # Prepare data for mir_eval
            ref_tempi, weight = prepare_truth(reference_tempi)
            est_tempi = prepare_preds(estimated_tempi)

            # Evaluate with ±8% tolerance
            p_score, _, _ = mir_eval.tempo.detection(
                ref_tempi,
                est_tempi,
                tolerance=0.08  # ±8% as specified in challenge rules
            )
        else:
            p_score = 0.

        sum_p += p_score
    return sum_p / len(target)


def check_size(path):
    """Check if file size is reasonable"""
    size = os.path.getsize(path)
    if size == 0 or size > 2 ** 24:
        raise RuntimeError(f'input file "{path}" '
                           'has weird size: "{}" [bytes]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    if args.submission is None or args.target is None:
        print(f'script needs two args: {args}')
        return -1

    check_size(args.submission)
    check_size(args.target)

    with open(args.submission, 'r') as fh:
        submission = json.load(fh)

    with open(args.target, 'r') as fh:
        target = json.load(fh)

    print(evaluate_loop(submission, target))


if __name__ == '__main__':
    main()