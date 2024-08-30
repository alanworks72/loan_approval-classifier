import joblib
from utils import loadConfig
from src.loader import loadBatches
from src.report import explainResults


def calcInterestRate(credit_score, dti_ratio, loan_amount, employment_status):
    # 기본 금리 설정
    base_rate = 4.0

    # 신용 점수에 따른 금리 조정
    if credit_score > 750:
        rate = base_rate - 1.0  # 신용 점수가 매우 높으면 낮은 금리 적용
    elif credit_score > 700:
        rate = base_rate - 0.5  # 신용 점수가 높으면 기본 금리보다 약간 낮음
    elif credit_score > 650:
        rate = base_rate  # 신용 점수가 평균 정도면 기본 금리 적용
    else:
        rate = base_rate + 1.0  # 신용 점수가 낮으면 높은 금리 적용

    # DTI 비율에 따른 금리 조정
    if dti_ratio > 0.4:
        rate += 1.0  # DTI가 0.4 초과시 금리 상승
    elif dti_ratio > 0.35:
        rate += 0.5  # DTI가 0.35 초과시 금리 약간 상승
    elif dti_ratio > 0.3:
        rate += 0.25  # DTI가 0.3 초과시 금리 소폭 상승
    
    # 고용 상태에 따른 금리 조정
    if employment_status == "Unemployed":
        rate += 1.5  # 실직 상태일 경우 높은 금리 적용
    elif employment_status == "Self-employed":
        rate += 0.5  # 자영업일 경우 다소 높은 금리 적용

    # 대출 금액에 따른 금리 조정
    if loan_amount > 50000:
        rate += 0.5  # 대출 금액이 50,000달러 초과시 금리 상승
    elif loan_amount > 100000:
        rate += 1.0  # 대출 금액이 100,000달러 초과시 금리 추가 상승
    
    return rate

def digitKorean(digit):
    digits = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    return digits[int(digit)]

def exchangeWon(num):
    num = int(num)

    units = ['', '만', '억', '조']
    num_str = str(num)
    length = len(num_str)

    # 네 자릿수씩 끊어서 처리
    parts = []
    while num_str:
        parts.insert(0, num_str[-4:])  # 뒤에서부터 4자리씩 잘라서 앞에 삽입
        num_str = num_str[:-4]

    result = []
    for i, part in enumerate(parts):
        if int(part) == 0:  # '0000' 등 0으로만 된 부분은 건너뜀
            continue

        part_result = []
        part_length = len(part)
        for j, digit in enumerate(part):
            if digit == '0':
                continue
            korean_digit = digitKorean(digit)
            if j == part_length - 1:  # 일의 자리
                part_result.append(f"{korean_digit}")
            elif j == part_length - 2:  # 십의 자리
                if digit != '1':  # '십' 앞에 '일'은 생략
                    part_result.append(f"{korean_digit}십")
                else:
                    part_result.append(f"십")
            elif j == part_length - 3:  # 백의 자리
                part_result.append(f"{korean_digit}백")
            elif j == part_length - 4:  # 천의 자리
                part_result.append(f"{korean_digit}천")

        result.append(''.join(part_result) + units[len(parts) - i - 1])

    return ''.join(result)

def calcTotalInterest(loan_amount, interest_rate, repayment_years, repayment_method):
    monthly_rate = interest_rate / 12 / 100  # 월 이율
    months = repayment_years * 12  # 총 상환 개월 수

    if repayment_method == "원리금 균등 상환":
        monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -months)
        total_payment = monthly_payment * months
        return total_payment - loan_amount, months, monthly_payment
    elif repayment_method == "원금 균등 상환":
        principal_payment = loan_amount / months
        total_payment = sum([(loan_amount - principal_payment * i) * monthly_rate for i in range(months)]) + loan_amount
        return total_payment - loan_amount, months, None  # 원금 균등 상환 방식에서는 월 상환 금액이 변동되므로 None 처리
    else:
        raise ValueError("지원되지 않는 상환 방식입니다.")

def run(config):
    # Batch 불러오기
    test_batches = loadBatches(config["test_file"], is_train=False)

    # 학습된 모델 불러와 추론
    model = joblib.load("model.pkl")
    pred = model.predict(test_batches[0])

    exchange_rate = 1330.21

    # 대출 결과 출력
    for idx in range(len(pred)):
        print("-"*40)
        if pred[idx] == 0:
            print("고객님의 대출신청이 거절되었습니다.")
            user_input = input("\n대출 심사 상세 내역을 확인 하시려면 'Y' 를, 대출신청을 종료하시려면 'N' 을 입력해주세요: ").strip().lower()

            if user_input == 'y':
                # 상세 내역 확인
                print("대출 심사 상세 내역을 확인합니다. 잠시만 기다려주세요.")
                explainResults(model, test_batches[0], idx)
            elif user_input == 'n':
                print("대출신청을 종료합니다.\n\n이용해주셔서 감사합니다.")
                continue
            else:
                print("잘못된 입력으로 인해 대출신청을 종료합니다.\n\n이용해주셔서 감사합니다.")
                continue
        else:
            # 후처리 필요항목 불러오기
            loan_amount = test_batches[0].iloc[idx]["Loan Amount"]
            max_loan_limit = test_batches[0].iloc[idx]["Maximum Loan Limit"]
            credit_score = test_batches[0].iloc[idx]["Credit Score"]
            dti_ratio = test_batches[0].iloc[idx]["Debt-to-Income Ratio"]
            employment_status = test_batches[0].iloc[idx]["Employment Status"]

            # 실제 승인된 대출 금액은 신청 금액과 최대 대출 한도 중 작은 값으로 설정
            approved_loan_amount = min(loan_amount, max_loan_limit)

            # 대출 금리 계산
            interest_rate = calcInterestRate(credit_score, dti_ratio, approved_loan_amount, employment_status)

            repayment_years = 5
            repayment_method = "원리금 균등 상환"

            # 대출 금액 산정 및 한글표기 생성
            approved_loan_amount = int(approved_loan_amount * exchange_rate)
            won = exchangeWon(approved_loan_amount)
            
            # 상환 총액 한글표기 생성
            total_interest, months, monthly_payment = calcTotalInterest(approved_loan_amount, interest_rate, repayment_years, repayment_method)
            total_won = exchangeWon(int(approved_loan_amount + total_interest))

            # 결과 출력
            print("고객님의 대출신청이 승인되었습니다.\n")
            print(f"대출 승인 금액: {approved_loan_amount:,}원(금 {won}원)")
            print(f"적용 금리: 연 {interest_rate:.2f}%,")
            print(f"대출 기간: {repayment_years}년 ({months}개월)")
            print(f"상환 방식: {repayment_method}\n")
            print(f"대출 상환 일정에 따라 '매월 1일' '지정된 계좌'로 부터 '{months}개월' 간 '{int(monthly_payment):,}원'이 '자동이체'됩니다.")
            print(f"상환 기간 중 납부 예정 총 이자 금액은 {int(total_interest):,}원 이며,\n총 상환 예정 금액은 {int(approved_loan_amount+total_interest):,}원(금 {total_won}원) 입니다.")
            print("\n이용해주셔서 감사합니다.")
            print("-" * 40)
            print("")


if __name__ == "__main__":
    config = loadConfig()
    run(config)