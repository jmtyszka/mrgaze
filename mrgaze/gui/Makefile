QTDIR=mrgaze_gui
QTUI=$(QTDIR)/mainwindow.ui
PYUI=../gui.py

all: $(PYUI)

$(PYUI) : $(QTUI)
	pyuic5 $(QTUI) -o $(PYUI)
