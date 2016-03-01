QTDIR=mrgaze/gui/MrGaze
QTUI=$(QTDIR)/mrgaze.ui
PYUI=mrgaze/gui.py

all: $(PYUI)

$(PYUI): $(QTUI)
	pyuic5 $(QTUI) -o $(PYUI)

clean:
	rm -rf $(PYUI)
