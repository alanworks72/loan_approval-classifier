import numpy as np
from shap.plots import waterfall
from shap.explainers import TreeExplainer

def explainResults(model, batch, index):
    # SHAP 알고리즘 초기화
    explainer = TreeExplainer(model)
    expected_value = explainer.expected_value[1]
    shap_values = explainer(batch, check_additivity=False)

    # SHAP 결과 분할
    contributions = shap_values[index][:, 1]
    features = batch.columns

    # 심사 점수 책정
    loan_score = abs((expected_value + np.sum(contributions.values))*100)
    
    print(f"\n=== 고객님의 대출 심사 점수는 {loan_score:.2f}점 입니다. ===\n")
    
    if loan_score < 50.:
        print("대출 심사 주요 항목 평가는 다음과 같습니다.\n")

        # 증감 Feature 분할
        positive_contributions = [(feature, contribution) for feature, contribution in zip(features, contributions.values) if contribution > 0]
        negative_contributions = [(feature, contribution) for feature, contribution in zip(features, contributions.values) if contribution < 0]

        # 주요 증감 Feature 추출
        top3_positives = sorted(positive_contributions, key=lambda x: -x[1])[:3]
        top3_negatives = sorted(negative_contributions, key=lambda x: x[1])[:3]

        # 결과 리포트 출력
        print("* 대출 가능성 상승 요인")
        for i in range(len(top3_positives)):
            feature, contribution = top3_positives[i]
            print(f"{i+1}. {feature}: {contribution*100:.2f}% 상승")
        print("\n* 대출 가능성 하락 요인")
        for i in range(len(top3_negatives)):
            feature, contribution = top3_negatives[i]
            print(f"{i+1}. {feature}: {contribution*100:.2f}% 하락")
    
        """ 증감 요인 상세 소개
        for feature, data, contribution in zip(features, contributions.data, contributions.values):
            if abs(contribution*100) > 0.01:
                print(f"* {feature} 요인이 대출 가능성을 {contribution*100:.2f}% {'증가' if contribution > 0 else '감소'}시켰습니다.")
        print("-"*40)
        print("")
        waterfall(shap_values[index][:,1], max_display=999, show=True)
        """
    print("\n대출 심사 상세 내역 확인 서비스를 종료합니다.\n\n이용해주셔서 감사합니다.")
    print("-"*40)
    print("")