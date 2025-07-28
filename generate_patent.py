import argparse
import os
import openai


def build_prompt(summary: str, num_claims: int) -> str:
    """Return a detailed prompt to generate a Korean patent specification."""

    return f"""
다음 특허 요약을 바탕으로 대한민국 특허출원 명세서를 작성해 주세요. 청구항은 총 {num_claims}개 작성합니다.

특허 요약:
{summary}

아래의 형식을 따라 상세하게 작성해 주세요.

1. 기술 분야
2. 배경기술
3. 발명의 내용
   3.1 해결하려는 과제
   3.2 과제의 해결수단
   3.3 발명의 효과
4. 청구항 (번호를 매겨 {num_claims}개 작성)
"""


def generate_spec(summary: str, num_claims: int) -> str:
    """Use the OpenAI API to create a patent specification."""

    prompt = build_prompt(summary, num_claims)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "당신은 대한민국 특허 명세서를 작성하는 전문가입니다.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Korean patent specification using OpenAI API")
    parser.add_argument("--summary", required=True, help="Patent summary text")
    parser.add_argument(
        "--claims",
        type=int,
        default=1,
        help="Number of claims to generate (default: 1)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    openai.api_key = api_key

    spec = generate_spec(args.summary, args.claims)
    print(spec)


if __name__ == "__main__":
    main()

