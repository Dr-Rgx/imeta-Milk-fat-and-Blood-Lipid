# coding=utf-8

def send_masaage(tel_nub,exp_name,server):

    def md5(str):
        import hashlib
        m = hashlib.md5()
        m.update(str.encode("utf8"))
        return m.hexdigest()


    statusStr = {
        '0': '短信发送成功',
        '-1': '参数不全',
        '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
        '30': '密码错误',
        '40': '账号不存在',
        '41': '余额不足',
        '42': '账户已过期',
        '43': 'IP地址限制',
        '50': '内容含有敏感词'
    }

    # temp = '你在服务器"{serve_name}"上的实验名称为"{exp_name}"，已经训练完成，请查看'


    def get_exp_str(serve_name, exp_name):
        print(f'【auroralab实验室】你在服务器"{serve_name}"上的实验名称为"{exp_name}"，已经训练完成，请查看')
        return f'【auroralab实验室】你在服务器"{serve_name}"上的实验名称为"{exp_name}"，已经训练完成，请查看'


    smsapi = "http://api.smsbao.com/"
    # smsapi = "http://api.smsbao.com/sms?u=USERNAME&p=PASSWORD&m=PHONE&c=CONTENT/"
    # 短信平台账号
    user = 'no_grad'
    # 短信平台密码
    # password = md5('nova1996')
    password = 'fa989fb8de224e6cbc140f4f38a82b46'
    # 要发送的短信内容
    content = get_exp_str(server, exp_name)
    # 要发送短信的手机号码
    phone = tel_nub
    # phone = '17702348466'
    # phone = '18553253958'

    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])


if __name__ == '__main__':

    import urllib
    import urllib.request
    import hashlib
    import argparse
    parser = argparse.ArgumentParser(description='send massage when code finish')
    parser.add_argument('--tel_nub', type=str,required=True)
    parser.add_argument('--exp_name', type=str,required=False,help='实验名字',default='实验')
    parser.add_argument('--server', type=str,required=True,help='服务器名字')

    args = parser.parse_args()

    send_masaage(args.tel_nub,args.exp_name,args.server)

    # print(args)
