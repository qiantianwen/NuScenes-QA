# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import json, pickle
import numpy as np
from collections import defaultdict


def Eval(__C, dataset, ans_ix_list, log_file, result_eval_file):
    
    ques_file_path = __C.RAW_PATH['val']
    true_answers = []
    predicted_answers = []
    qu = []
    tokens = []
    with open(ques_file_path, 'r') as f:
        questions = json.load(f)['questions']
        
        for ix, ques in enumerate(questions):
            qu.append(ques['question'])
            tokens.append(ques['sample_token'])
            true_answers.append(str(ques['answer']))
            predicted_answers.append(dataset.ix2ans[str(ans_ix_list[ix])])
    
    correct_by_q_type = defaultdict(list)

    
    num_true, num_pred = len(true_answers), len(predicted_answers)
    assert num_true == num_pred, 'Expected %d answers but got %d' % (
        num_true, num_pred)
    
    for i, (true_answer, predicted_answer) in enumerate(zip(true_answers, predicted_answers)):
        correct = 1 if true_answer == predicted_answer else 0
        correct_by_q_type['Overall'].append(correct)
        q_type = questions[i]['template_type']
        sub_q_type = q_type + '_' + str(questions[i]['num_hop'])
        correct_by_q_type[q_type].append(correct)
        correct_by_q_type[sub_q_type].append(correct)
    
    print('Write to log file: {}'.format(log_file))
    logfile = open(log_file, 'a+')
    q_dict = {}
    for q_type, vals in sorted(correct_by_q_type.items()):
        vals = np.asarray(vals)
        q_dict[q_type] = [vals.sum(), vals.shape[0]]
    for q_type in q_dict:
        val, tol = q_dict[q_type]
        print(q_type, '%d / %d = %.2f' % (val, tol, 100.0 * val / tol))
        logfile.write(q_type + ' : ' + '%d / %d = %.2f\n' % (val, tol, 100.0 * val / tol))
    
    logfile.write("\n")
    logfile.close()

    if result_eval_file is not None:
        print('Write prediction to result file: {}'.format(result_eval_file))
        result_fs = open(result_eval_file, 'w')
        for i, (token, que, pred, gt) in enumerate(zip(tokens, qu, predicted_answers, true_answers)):
            result_fs.write(token)
            result_fs.write("    ")
            result_fs.write(que)
            result_fs.write("    ")
            result_fs.write(pred)
            result_fs.write("    ")
            result_fs.write(gt)
            result_fs.write("\n")
        result_fs.close()
        print('Finished!')

