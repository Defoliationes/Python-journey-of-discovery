import time
import pickle


def simple_progress_bar(total):
    bar_length = 50
    for i in range(total):
        percent = float(i + 1) / total
        filled_length = int(bar_length * percent)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r请稍等: |{bar}| {percent * 100:.2f}%', end='')
        time.sleep(0.1)
    print()


def enroll(users):
    file_path = 'ID.txt'
    identity_phone()
    users = user(file_path, users)
    return users


def identity_phone():
    name = str(input("请输入你的名字："))
    while True:
        identityID = str(input("请输入你的身份证号："))
        if len(identityID) != 18:
            print('请输入正确的身份证号!!')
        else:
            while True:
                phone_number = str(input("请输入你的手机号码："))
                if len(phone_number) != 11:
                    print('请输入正确的手机号！！')
                else:
                    print('请继续注册！！')
                    break
            break
            return


def landing(users):
    while True:
        NumberID = str(input('请输入你的账户：'))
        password = int(input('请输入你的密码：'))
        if users[NumberID] == password:
            simple_progress_bar(50)
            print('登陆成功！！')
            break
        else:
            print('密码错误，请重新输入或退出！！')
            reply = str(input('是否退出：'))
            if reply == '是':
                break
            else:
                print('请重新输入!!')
    return NumberID


def operate(NumberID, users):
    print('晚上好，点宽用户!')
    while True:
        print('您想做什么呢?\n1-检查余额\n2-存款\n3-取款\n4向其他账户转账\n5-退出登录')
        num = eval(input('请输入：'))
        if num == 1:
            balance = users[NumberID + '_balance']
            print(f'您当前的余额为：{balance}')
        if num == 2:
            recorded = float(input('请输入您要存款的金额：'))
            users[NumberID + '_balance'] = users[NumberID + '_balance'] + recorded
            balance = users[NumberID + '_balance']
            simple_progress_bar(50)
            print(f'您当前的余额为：{balance}')
        elif num == 3:
            while True:
                billing = float(input('请输入您要取款的金额：'))
                balance = users[NumberID + '_balance']
                if balance < billing:
                    print('抱歉，您的金额不足！')
                    reply = str(input('是否返回：'))
                    if reply == '否':
                        print('请重新输入!!')
                    else:
                        break
                else:
                    balance = balance - billing
                    users[NumberID + '_balance'] = balance
                    simple_progress_bar(50)
                    print(f'您当前的余额为：{balance}')
                    break
        elif num == 4:
            while True:
                another_ID = str(input('请输入您要转账的用户 ID：'))
                simple_progress_bar(50)
                if another_ID not in users:
                    print('抱歉，您要转账的用户 ID 不存在。')
                    reply = str(input('是否返回：'))
                    if reply == '否':
                        print('请重新输入')
                    else:
                        break
                else:
                    while True:
                        another_balance = float(input('请输入您要转账的金额：'))
                        balance = users[NumberID + '_balance']
                        if balance < another_balance:
                            print('转账失败，您的金额不足！')
                            reply = str(input('是否返回：'))
                            if reply == '否':
                                print('请重新输入!!')
                            else:
                                break
                        else:
                            reply = str(input('是否确认转账：'))
                            if reply == '是':
                                simple_progress_bar(50)
                                balance = balance - another_balance
                                users[another_ID + '_balance'] += another_balance
                                print(f'转账成功，您当前的余额为：{balance}')
                                break
                            else:
                                print('请重新输入!!')
                    break
        elif num == 5:
            with open('Bank_users.pkl', 'wb') as file:
                pickle.dump(users, file)
            break
            print('账户已退出——')
            return


def user(file_path, users):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = int(content)
    while True:
        password_1 = int(input("请输入你的密码："))
        if len(str(password_1)) != 6:
            print('密码只能是6位，请重新输入！！')
        else:
            break

    while True:
        password_2 = int(input("请确认你的密码："))
        if password_1 != password_2:
            print('输入不一致，请重新输入！！')
        else:
            simple_progress_bar(50)
            print(f'你的账户是{content}')
            print(f'你的密码是{password_1}')
            break
    NumberID = str(content + 1)
    content = str(content)
    users[content] = password_1
    users[content + '_balance'] = 0

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        content = content.replace(content, NumberID)
        file.write(content)
    with open('Bank_users.pkl', 'wb') as f:
        pickle.dump(users, f)
    return users


def welcome():
    print('欢迎来到点宽商业银行，请问您今天想做什么? ')
    while True:
        print('您想做什么呢？\n1-创建账户\n2-登陆账户\n3-退出')
        num = eval(input('请输入：'))
        if num == 1:
            with open('Bank_users.pkl', 'rb') as file:
                users = pickle.load(file)
            users = enroll(users)

        elif num == 2:
            with open('Bank_users.pkl', 'rb') as file:
                users = pickle.load(file)
            NumberID = landing(users)
            operate(NumberID, users)
            simple_progress_bar(50)
        elif num == 3:
            print('谢谢您的使用，再见！')
            break
        else:
            print('输出错误，请重新输入！！')
    return users


if __name__ == "__main__":
    welcome()