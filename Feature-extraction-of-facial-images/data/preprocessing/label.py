# 定义特征映射表
feature_mapping = {
    '_sex': {'male': 0, 'female': 1},
    '_age': {'child': 0, 'teen': 1, 'adult': 2, 'senior': 3},
    '_race': {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4},
    '_face': {'smiling': 0, 'serious': 1, 'funny': 2},
}


def parse_features(feature_string):
    """解析一行特征字符串为键值对"""
    features = feature_string.split(') (')
    features[0] = features[0][1:]  # 去掉第一个 '('
    features[-1] = features[-1][:-1]  # 去掉最后一个 ')'

    feature_values = {}
    for feature in features:
        key, value = feature.split()
        feature_values[key] = value
    return feature_values


def encode_label(features, mapping):
    """将特征转换为唯一数字标签"""
    label = 0
    base = 1
    for key in reversed(list(mapping.keys())):  # 从最后一个特征开始编码
        label += mapping[key][features[key]] * base
        base *= len(mapping[key])
    return label



def process_file(input_file, output_file, mapping):
    """将文件中每行数据转换为数字标签"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            splits = line.split(maxsplit=1)
            if len(splits) != 2:
                continue
            index, feature_string = splits
            feature_values = parse_features(feature_string)
            label = encode_label(feature_values, mapping)
            outfile.write(f"{index} {label}\n")


# 示例调用
input_file = 'faceDR.txt'  # 输入文件
output_file = 'labels.txt'  # 输出文件
process_file(input_file, output_file, feature_mapping)
