###TWILIO MESSAGE CODE###
##code for sending messages to single phone number##

import os
from twilio.rest import Client

account_sid = 'AC62722f2f1d0810d0868c19d599c21969'
auth_token = '9559ef912bc58b3d277cec38d0ea21cd'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Hey This is Sindhu.Bye!",
                     from_='+16467988374',
                     to='+917010443489'
                 )

print(message.sid)


##########################################################################################

##code for sending messages to multiple phone numbers##
ccount_sid = 'AC62722f2f1d0810d0868c19d599c21969'
auth_token = '9559ef912bc58b3d277cec38d0ea21cd'
client = Client(account_sid, auth_token)

numbers_to_message = ['+917010443489', '+918970370853', '+918217606386','+919611621675']
for number in numbers_to_message:
    client.messages \
        .create(
        body = 'Hello from my Twilio number!',
        from_ = '+16467988374',
        to = number
    )
