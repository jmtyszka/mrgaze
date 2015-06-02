PUPIL_SAMPLES = Samples/Pupil.vec
PUPIL_CASCADE = Cascade

NUM_NEG = 14
NUM_POS = 200

all: samples train

samples : $(PUPIL_SAMPLES)

$(PUPIL_SAMPLES):
	opencv_createsamples -vec $(PUPIL_SAMPLES) -info Positive.txt -bg Negative.txt -num $(NUM_POS)

train: $(PUPIL_SAMPLES)
	opencv_traincascade -data $(PUPIL_CASCADE) -vec $(PUPIL_SAMPLES) -bg Negative.txt -numPos $(NUM_POS) -numNeg $(NUM_NEG)

clean:
	rm -rf $(PUPIL_SAMPLES) ${PUPIL_CASCADE}/*
