QTDIR=mrgaze/gui/MrGaze
QTUI=$(QTDIR)/mrgaze.ui
QTRC=$(QTDIR)/icons.qrc
PYUI=mrgaze/qtui.py
PYRC=mrgaze/icons_rc.py

all: $(PYUI) $(PYRC)

install:
	sudo python setup.py install

# UI conversion
$(PYUI): $(QTUI)
	pyuic5 $(QTUI) -o $(PYUI)

# UI resource conversion
$(PYRC): $(QTRC)
	pyrcc5 $(QTRC) -o $(PYRC)

clean:
	rm -rf $(PYUI) $(PYRC)
