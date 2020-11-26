import os
import time
from time import sleep
from pynq.gpio import GPIO

def distance(measure):
    try:
        # Find the GPIO base pin
        GPIO_MIN_USER_PIN = 0
        index = 0
        for root, dirs, files in os.walk('/sys/class/gpio'):
            for name in dirs:
                if 'gpiochip' in name:
                   index = int(''.join(x for x in name if x.isdigit()))
                   break
        base = GPIO.get_gpio_base()
        # print(base)
        # print(index)
        assert base == index, 'GPIO base not parsed correctly.'
        ps_mio36 = base + 36
        ps_mio44 = base + 44
        trig = GPIO(ps_mio36, 'out')
        echo = GPIO(ps_mio44, 'in')
        trig.write(0)
        sleep(0.002)
        trig.write(1)
        sleep(0.010)
        trig.write(0)
        while echo.read() == 0:
            nosig = time.time()
        #print(nosig)
        nosig = 1000 * (round(nosig, 9) - int(nosig))
        #print(nosig)
        nosig = 1000 * nosig
        #print(nosig)
        nosig = int(round(nosig, 0))
        #print(nosig)
        while echo.read() == 1:
            sig = time.time()
        #print(sig)
        sig = 1000 * (round(sig, 9) - int(sig))
        #print(sig)
        sig = 1000 * sig
        #print(sig)
        sig = int(round(sig, 0))
        #print(sig)
        #print(nosig)
        t1 = sig - nosig
        #print(t1)
        if measure == 'cm':
            distance = round(t1/58, 1)
        elif measure == 'in':
            distance = round(tl/148, 1)
        else:
            print('improper choice of measurement: in or cm')
            distance = None

        del trig
        del echo
        return distance

    except:
        distance = 100
        return distance

if __name__ == "__main__":
    while True:
        print(str(distance('cm'))+' cm')
        if distance('cm') < 15:
            break
        else:
            # delay for 1 second
            sleep(1.0)
