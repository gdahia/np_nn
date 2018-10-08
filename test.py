import argparse


def read_gt(path):
  gt = {}
  with open(path, 'r') as f:
    for line in f:
      orig, name = line.strip().split(':')
      label = orig.split('/')[0]
      gt[name] = label

  return gt


def read_preds(path):
  preds = {}
  with open(path, 'r') as f:
    for line in f:
      name, label = line.strip().split()
      name = name.strip('.png')
      preds[name] = label

  return preds


def diff_accuracy(gt, preds):
  correct = 0
  for name, label in preds.items():
    if name in gt and gt[name] == label:
      correct += 1

  total = len(gt)

  return correct / total


def main(gt_path, preds_path):
  gt = read_gt(gt_path)
  preds = read_preds(preds_path)
  print(diff_accuracy(gt, preds))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('gt_path', type=str, help='path to ground truth file')
  parser.add_argument('preds_path', type=str, help='path to predictions file')

  flags = parser.parse_args()

  main(flags.gt_path, flags.preds_path)
