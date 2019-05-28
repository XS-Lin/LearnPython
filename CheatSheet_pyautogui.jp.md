
# Cheat Sheet #

pyautogui.PAUSE = 2.5 # TODO
pyautogui.FAILSAFE = True # TODO

## General Functions ##

~~~python
pyautogui.position()  
#   Point(x=1017, y=321)

pyautogui.size()
#   Size(width=1920, height=1080)

pyautogui.onScreen(x, y) # True if x & y are within the screen.
#   True
~~~

## Mouse Functions ##

~~~python
pyautogui.moveTo(x, y, duration=num_seconds)
pyautogui.moveRel(xOffset, yOffset, duration=num_seconds)

pyautogui.dragTo(x, y, duration=num_seconds)
pyautogui.dragRel(xOffset, yOffset, duration=num_seconds)

pyautogui.click(x=moveToX, y=moveToY, clicks=num_of_clicks, interval=secs_between_clicks, button='left') #'left','middle','right'
pyautogui.rightClick(x=moveToX, y=moveToY)
pyautogui.middleClick(x=moveToX, y=moveToY)
pyautogui.doubleClick(x=moveToX, y=moveToY)
pyautogui.tripleClick(x=moveToX, y=moveToY)

pyautogui.scroll(amount_to_scroll, x=moveToX, y=moveToY)
pyautogui.mouseDown(x=moveToX, y=moveToY, button='left')
pyautogui.mouseUp(x=moveToX, y=moveToY, button='left')
~~~

## Keyboard Functions ##

~~~python
pyautogui.typewrite('Hello world!\n', interval=secs_between_keys)  # useful for entering text, newline is Enter
pyautogui.typewrite(['a', 'b', 'c', 'left', 'backspace', 'enter', 'f1'], interval=secs_between_keys)
pyautogui.hotkey('ctrl', 'c')  # ctrl-c to copy
pyautogui.hotkey('ctrl', 'v')  # ctrl-v to paste

pyautogui.keyDown(key_name)
pyautogui.keyUp(key_name)

pyautogui.alert('This displays some text with an OK button.') 
pyautogui.confirm('This displays text and has an OK and Cancel button.')
#   'OK'

pyautogui.prompt('This lets the user type in a string and press OK.')
#   'This is what I typed in.'
~~~

## Screenshot Functions ## 

~~~python
pyautogui.screenshot() 
#   <PIL.Image.Image image mode=RGB size=1920x1080 at 0x24AE974B710>

pyautogui.screenshot('foo.png') # returns a Pillow/PIL Image object, and saves it to a file
#   <PIL.Image.Image image mode=RGB size=1920x1080 at 0x31AA198>
#
pyautogui.locateOnScreen('looksLikeThis.png')  # returns (left, top, width, height) of first place it is found
#   (863, 417, 70, 13)

for i in pyautogui.locateAllOnScreen('looksLikeThis.png')
    list(pyautogui.locateAllOnScreen('looksLikeThis.png'))
    pyautogui.locateCenterOnScreen('looksLikeThis.png')  # returns center x and y
#   (898, 423)
#   ...
#   These functions return None if the image couldnâ€™t be found on the screen.
#   Note: The locate functions are slow and can take a full second or two.
~~~
