from gqa_eval import GQAEval
from plot_tail import plot_tail
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--eval_tail_size',  action='store_true')
parser.add_argument('--ood_test',  action='store_true')
parser.add_argument('--predictions',  type=str)
args = parser.parse_args()


if args.eval_tail_size:
    result_eval_file = args.predictions

    # Retrieve scores
    alpha_list = [9.0, 7.0, 5.0, 3.6, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8, 0.4, 0.3, 0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6, -0.7]
    acc_list = []
    for alpha in alpha_list:
    
        ques_file_path = '../data/alpha_tail/val_bal_tail_%.1f.json' % alpha
        
        gqa_eval = GQAEval(result_eval_file, ques_file_path, choices_path=None, EVAL_CONSISTENCY=False)
        acc = gqa_eval.get_acc_result()['accuracy'] 
        acc_list.append(acc)

    print("Alpha:", alpha_list)
    print("Accuracy:", acc_list)
    # Plot: save to "tail_plot_[model_name].pdf"
    plot_tail(alpha=list(map(lambda x: x+1, alpha_list)), accuracy=acc_list, model_name='default')  # We plot 1+alpha vs. accuracy
elif args.ood_test:
    result_eval_file = args.predictions
    file_list = {'Tail': 'ood_testdev_tail.json', 'Head': 'ood_testdev_head.json', 'All': 'ood_testdev_all.json'}
    result = {}
    for setup, ques_file_path in file_list.items(): 
        gqa_eval = GQAEval(result_eval_file, '../data/'+ques_file_path, choices_path=None, EVAL_CONSISTENCY=False)
        result[setup] = gqa_eval.get_acc_result()['accuracy']

        result_string, detail_result_string = gqa_eval.get_str_result()
        print('\n___%s___'%setup)
        for result_string_ in result_string:
            print(result_string_)

    print('\nRESULTS:\n')
    msg = 'Accuracy (tail, head, all): %.2f, %.2f, %.2f' % (result['Tail'], result['Head'], result['All'])
    print(msg)
