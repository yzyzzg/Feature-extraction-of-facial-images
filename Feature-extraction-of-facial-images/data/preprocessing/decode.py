# 定义特征映射表
feature_mapping = {
    '_face': {'smiling': 0, 'serious': 1, 'funny': 2},
    '_race': {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4},
    '_age': {'child': 0, 'teen': 1, 'adult': 2, 'senior': 3},
    '_sex': {'male': 0, 'female': 1},
}


def decode_label(label, mapping):
    """根据数字标签逆推特征值"""
    decoded_features = {}
    base = 1
    for key in mapping.keys():
        base *= len(mapping[key])

    for key in reversed(list(mapping.keys())):  # 反向解码，从最后一个特征开始
        base //= len(mapping[key])
        value_index = label // base
        label %= base
        for value, index in mapping[key].items():
            if index == value_index:
                decoded_features[key] = value
                break
    return decoded_features


def main():
    """主函数：输入标签并输出特征"""
    while True:
        try:
            # 提示用户输入标签
            label = int(input("请输入数字标签 (输入负数退出)："))
            if label < 0:
                print("程序已退出。")
                break

            # 计算特征
            features = decode_label(label, feature_mapping)
            print("对应的特征：")
            for key, value in features.items():
                print(f"{key}: {value}")
        except ValueError:
            print("输入无效，请输入一个有效的整数。")
        except Exception as e:
            print(f"发生错误：{e}")


# 运行程序
if __name__ == "__main__":
    main()
