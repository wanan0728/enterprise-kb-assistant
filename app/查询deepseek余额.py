import requests

api_key = "sk-f359d11bb18e4d00b05b735d394f770e"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 尝试不同的端点
endpoints = [
    "https://api.deepseek.com/billing/usage",
    "https://api.deepseek.com/v1/billing/usage",
    "https://api.deepseek.com/user/balance",
    "https://api.deepseek.com/dashboard/billing/usage",
]

for endpoint in endpoints:
    print(f"\n尝试端点: {endpoint}")
    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"成功! 响应: {data}")
                break
            except:
                print(f"响应不是JSON: {response.text[:200]}")
        else:
            print(f"失败: {response.text[:200]}")
    except Exception as e:
        print(f"请求异常: {e}")