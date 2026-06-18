JAVAC = javac
JAVA = java
OUT_DIR = out

all: nn mlp svm svm-centroid voting

nn:
	mkdir -p $(OUT_DIR)/nn
	$(JAVAC) -d $(OUT_DIR)/nn algorithms/nearest-neighbor/NN_euclidean_dsitance.java

mlp:
	mkdir -p $(OUT_DIR)/mlp
	$(JAVAC) -d $(OUT_DIR)/mlp algorithms/mlp/MLP.java

svm:
	mkdir -p $(OUT_DIR)/svm
	$(JAVAC) -d $(OUT_DIR)/svm algorithms/svm/SVM.java

svm-centroid:
	mkdir -p $(OUT_DIR)/svm-centroid
	$(JAVAC) -d $(OUT_DIR)/svm-centroid algorithms/svm-centroid-kernel/SVM_centroid_kernel.java

voting:
	mkdir -p $(OUT_DIR)/voting
	$(JAVAC) -d $(OUT_DIR)/voting algorithms/voting-system/voting_system.java

run-nn: nn
	cd data && $(JAVA) -cp ../$(OUT_DIR)/nn digit_reco_NN.NN_euclidean_dsitance

run-mlp: mlp
	cd data && $(JAVA) -cp ../$(OUT_DIR)/mlp digit_reco_MLP.MLP

run-svm: svm
	cd data && $(JAVA) -cp ../$(OUT_DIR)/svm digit_reco_SVM.SVM

run-svm-centroid: svm-centroid
	cd data && $(JAVA) -cp ../$(OUT_DIR)/svm-centroid SVM_centroid_kernel

run-voting: voting
	cd data && $(JAVA) -cp ../$(OUT_DIR)/voting voting_system.voting_system

clean:
	rm -rf $(OUT_DIR)
