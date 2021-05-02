###TWILIO MESSAGE CODE###
##code for sending messages to single phone number##

import os
from twilio.rest import Client
account_sid = 'AC6xxxxxxxxxxxxxxxxxxxxxxxxxxxxx69'
auth_token = '95xxxxxxxxxxxxxxxxxxxxxxxxxxxxxcd'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Hey This is Sindhu.Bye!",
                     from_='+16xxxxxxxxx',
                     to='+91xxxxxxxxxx'
                 )

print(message.sid)


##########################################################################################

##code for sending messages to multiple phone numbers##
import os
from twilio.rest import Client
ccount_sid = 'AC6xxxxxxxxxxxxxxxxxxxxxxxxxxxxx69'
auth_token = '95xxxxxxxxxxxxxxxxxxxxxxxxxxxxxcd'
client = Client(account_sid, auth_token)

numbers_to_message = ['+91xxxxxxxxxx','+91xxxxxxxxxx','+91xxxxxxxxxx','+91xxxxxxxxxx']
for number in numbers_to_message:
    client.messages \
        .create(
        body = 'Hello from my Twilio number!',
        from_ = '+16xxxxxxxxx',
        to = number
    )
