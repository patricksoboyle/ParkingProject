import smtplib
import time

class notification:

    @staticmethod
    def send(contents):
        gmail_user = "wkucarcounting@gmail.com"
        gmail_password = "carcounter11"

        sent_from = gmail_user
        #add another string to the list for multiple recipients
        to = ["wkucarcounting@gmail.com"]
        subject = "Car Count Log %s" % (time.strftime("%D, %T"))

        email_text = """\  
        From: %s  
        To: %s  
        Subject: %s

        %s
        """ % (sent_from, ", ".join(to), subject, contents)

        try:  
            #establishes secure connection
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            #login and send email
            server.login(gmail_user, gmail_password)
            server.sendmail(sent_from, to, email_text)
            server.close()

            print('Count log sent')
        except:  
            print('Notification failed to send')
