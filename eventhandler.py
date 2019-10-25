# logging data and notifications
import time
import threading
from notificationsys import notification
# calling scripts
import subprocess

# USAGE
# Initializing an eventhandler:
#   lot_name = string name of parking lot
#   Optional:
#   notif_time = the period in hours between emails of the log file
#       notif_time is optional. If undefined, emails will be sent
#       for every event

class eventhandler():
    """Handles count data for a parking lot"""

    def __init__(self, lot_name, notif_time=0):
        """Start an event handler for a parking lot"""
        self.name = lot_name
        self.notif_time = notif_time
        start_count = 0

        # initialize daily log file if it isn't present
        subprocess.run(["bash", "ologhandler"])
        # store log for later reading
        self.log = "occupancylogs/" + time.strftime("%m-%d-%y") + ".log"

        # get the last line in the log file
        recent_line = self.getRecentLine()

        # if the most recent line is non-empty, then process the current count
        if recent_line:
            count, timestamp = recent_line.split()
            self.count = int(count)
        else:
            # the log is empty so the initial count will be set to the defined
            # starting count
            self.count = start_count
            # register the initial count in the log file
            self.registerEvent()

        # start emailing the daily log periodically if notif_time is set
        if notif_time != 0:
            # convert notification period from hours to seconds for use with
            # time.delay
            self.notif_time = notif_time * 60 * 60

            notifThread = threading.Thread(target=self.sendLog, args=(
                self.notif_time,))
            # start thread
            notifThread.start()

    def entry(self):
        """When a car enters the lot"""
        self.count += 1
        # add event info to log file
        self.registerEvent()
        # send email if notif_time isn't set
        if self.notif_time == None:
            notification.send("Entry\n" + self.getRecentLine())

    def exit(self):
        """When a car exits the lot"""
        self.count -= 1
        # add event info to log file
        self.registerEvent()
        # send email if notif_time isn't set
        if self.notif_time == None:
            notification.send("Exit\n" + self.getRecentLine())

    def manualSet(self, new_count):
        """Manually set the parking lot occupancy count and have this event
        marked """
        self.inputNewCount(new_count)
        self.appendLog("Manual Count Set")
        self.registerEvent()

    def askManualSet(self):
        """Ask for user input for manually setting the new parking lot count"""
        new_count = input("Set Count: ")
        self.manualSet(new_count)
        
    def inputNewCount(self, new_count):
        """Validate user input for new parking lot count"""
        try:
            new_count = int(new_count)
            if(new_count < 0):
                raise ValueError
            #change current count once new_count is validated
            self.count = new_count
        except:
            print("Error: New count must be an integer greater than or equal to 0")

    def appendLog(self, content):
        """Appends the daily log file with variable content"""
        with open(self.log, 'a') as file_object:
            file_object.write(content + "\n")

    def registerEvent(self):
        """Adds current count and timestamp to daily log"""
        self.appendLog(str(self.count) + " " + time.strftime("%H:%M"))

    def getLogLines(self):
        """Read log and return list of lines"""
        # read log
        with open(self.log) as file_object:
            contents = file_object.readlines()
        return contents

    def getLogContent(self):
        """Read log and return a string of the contents"""
        lines = self.getLogLines()
        content = ""
        # build string
        for line in lines:
            content = content + line
        return content

    def getRecentLine(self):
        """Get the last line in the log file"""
        try:
            log_content = self.getLogLines()
            # return last line
            return log_content[-1]
        # if it's empty, return None instead
        except IndexError:
            return None

    def sendLog(self, delay):
        """Send periodic log every x minutes (delay) to wkucarcounting@gmail.com"""
        while(True):
            time.sleep(delay)
            # read updated log
            content = self.getLogContent()
            # send email
            notification.send(content)
