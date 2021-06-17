import json
import linecache


def check(submit_path, test_path, max_num=15000):
    '''
    :param submit_path: 选手提交的文件名
    :param test_path: 原始测试数据名
    :param max_num: 测试数据大小
    :return:
    '''
    N = 0
    with open(submit_path, 'r', encoding='utf-8') as fs:
        for line in fs:
            try:
                line = line.strip()
                if line == '':
                    continue
                N += 1
                smt_jsl = json.loads(line)
                test_jsl = json.loads(linecache.getline(test_path, N))
                if set(smt_jsl.keys()) != {'text_id', 'query', 'candidate'}:
                    raise AssertionError(f'请保证提交的JSON数据的所有key与测评数据一致！ {line}')
                elif smt_jsl['text_id'] != test_jsl['text_id']:
                    raise AssertionError(f'请保证text_id和测评数据一致，并且不要改变数据顺序！text_id: {smt_jsl["text_id"]}')
                elif smt_jsl['query'] != test_jsl['query']:
                    raise AssertionError(f'请保证query内容和测评数据一致！text_id: {smt_jsl["text_id"]}, query: {smt_jsl["query"]}')
                elif len(smt_jsl['candidate']) != len(test_jsl['candidate']):
                    raise AssertionError(f'请保证candidate的数量和测评数据一致！text_id: {smt_jsl["text_id"]}')
                else:
                    for smt_cand, test_cand in zip(smt_jsl['candidate'], test_jsl['candidate']):
                        if smt_cand['text'] != test_cand['text']:
                            raise AssertionError(f'请保证candidate的内容和顺序与测评数据一致！text_id: {smt_jsl["text_id"]}, candidate_item: {smt_cand["text"]}')

            except json.decoder.JSONDecodeError:
                raise AssertionError(f'请检查提交数据的JSON格式是否有误！如：用键-值用双引号 {line}')
            except KeyError:
                raise AssertionError(f'请保证提交的JSON数据的所有key与测评数据一致！{line}')

    if N != max_num:
        raise AssertionError(f"请保证测试数据的完整性(共{max_num}条)，不可丢失或增加数据！")

    print('Well Done ！！')

check('柯哀无敌_ addr_parsing_runid.txt', 'data/final_test.txt')