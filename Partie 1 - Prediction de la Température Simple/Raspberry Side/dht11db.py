import RPi.GPIO as GPIO
import dht11
import time
import datetime
import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","root","password","Weather" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# initialize GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.cleanup()

# read data using pin GPIO4
instance = dht11.DHT11(pin=4)

while True:
    result = instance.read()
    if result.is_valid():
	now = datetime.datetime.now()
        t = now.strftime("%H-%M-%S")
	data = result.temperature
        print("Last valid input: " + str(t))
        print("Temperature: %d C" % data)

	# Prepare SQL query to INSERT a record into the database.
        sql = "INSERT INTO temp(time, value) VALUES ('%s', '%s')" % (t,data)
        try:
           # Execute the SQL command
           cursor.execute(sql)
           # Commit your changes in the database
           db.commit()
           print("Data saved .. ")
        except:
           # Rollback in case there is any error
           db.rollback()
           print("Data not saved !!")


    time.sleep(1)

# disconnect from server
db.close()
