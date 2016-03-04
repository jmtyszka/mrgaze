QTDIR=mrgaze/gui/MrGaze
QTUI=$(QTDIR)/mrgaze.ui
QTRC=$(QTDIR)/icons.qrc
PYUI=mrgaze/gui.py
PYRC=mrgaze/icons_rc.py

all: $(PYUI) $(PYRC) $(PYICONS)

install:
	sudo python setup.py install

# UI conversion
$(PYUI): $(QTUI)
	pyuic5 $(QTUI) -o $(PYUI)

# UI resource conversion
$(PYRC): $(QTRC)
	pyrcc5 $(QTRC) -o $(PYRC)

clean:
	rm -rf $(PYUI)
