import sys
import os
import json

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QDialog, QMainWindow, QMessageBox)
from PyQt5.uic import loadUi
from ui.main_window_ui import Ui_MainWindow
from ui.knn_ui import Ui_DialogKnn
from ui.knn_1_vs_rest_ui import Ui_DialogKnn1vsRest
import normalization
from scipy.spatial import distance as dis
import numpy as np


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.lines = []
        self.config = None
        self.connectSignalsSlots()

        self.normalized_path = None

    def connectSignalsSlots(self):
        self.pushButtonLoad.clicked.connect(self.load_file)
        self.pushButtonNormalize.clicked.connect(self.normalize)
        self.actionKnnTypeOne.triggered.connect(self.knn_type_one)
        self.actionKnnTypeTwo.triggered.connect(self.knn_type_two)
        self.actionSaveJson.triggered.connect(self.save_config_to_json)
        self.actionLoadJson.triggered.connect(self.load_config_from_json)
        self.actionReset.triggered.connect(self.reset_config)
        self.actionKnn1vsOther.triggered.connect(self.knn_1_vs_rest)

    def load_file(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a data file",
            directory=os.getcwd()
        )
        if response[0] != '':
            try:
                with open(response[0]) as file:
                    self.lines.clear()
                    splited_file = file.read().split("\n")

                    if self.lineEditSeparator.text() == "spacja":
                        separator = " "
                    else:
                        separator = self.lineEditSeparator.text()

                    for i in range(len(splited_file) - 1):
                        self.lines.append(normalization.Line(
                            splited_file[i], separator))

                    self.pushButtonNormalize.setEnabled(True)

                    self.lineEditLoad.clear()
                    self.lineEditLoad.insert(response[0])

            except:
                self.lineEditLoad.clear()
                self.lineEditLoad.insert("Błąd wczytywania")
                self.pushButtonNormalize.setEnabled(False)

            self.comboBoxDecisionColumn.clear()
            for i in range(len(self.lines[0].values)+1):
                self.comboBoxDecisionColumn.addItem(str(i))

            self.reset_config()
            self.actionSaveJson.setEnabled(False)

    def normalize(self):
        normalization.Line.SetDecision(
            self.lines, self.comboBoxDecisionColumn.currentText())

        if(self.config == None):
            normalization.Line.ConvertSymbolicToNumeric(
                normalization.Line, self.lines)
            normalization.Line.FindMinMaxInNumeric(self.lines)
        else:
            for line in self.lines:
                line.cnf = self.config

        normalization.Line.Normalization(
            self.lines, self.doubleSpinBoxMin.value(), self.doubleSpinBoxMax.value())

        response = QFileDialog.getSaveFileName(
            parent=self,
            caption="Select a data file",
            directory="normalized.dat"
        )
        if response[0] != '':
            with open(response[0], "w") as file:
                for line in self.lines:
                    for i in range(len(line.values)):

                        if i == len(line.values) - 1:
                            if i == line.cnf.decisionID:
                                file.write(str(line.decision))
                                break
                            else:
                                file.write(str(line.normalized[i]))
                                break
                        if i == line.cnf.decisionID:
                            file.write(str(line.decision) + ' ')
                        else:
                            file.write(str(line.normalized[i]) + ' ')

                    file.write("\n")

        self.normalized_path = response[0]
        if self.config == None:
            self.actionSaveJson.setEnabled(True)
            self.config = self.lines[0].cnf
        self.lineEditLoad.insert("Po wczytaniu nowego pliku zresetuje config")

    def knn_type_one(self):
        if self.normalized_path != None and self.lines[0].decision != None and len(self.lines[0].normalized) != 0:
            dialog_knn = DialogKnn(
                "one", self, self.config, self.lines, self.normalized_path)
        else:
            dialog_knn = DialogKnn("one", self, self.config)

        dialog_knn.exec()

    def knn_type_two(self):
        if self.normalized_path != None and self.lines[0].decision != None and len(self.lines[0].normalized) != 0:
            dialog_knn = DialogKnn(
                "two", self, self.config, self.lines, self.normalized_path)
        else:
            dialog_knn = DialogKnn("two", self, self.config)

        dialog_knn.exec()

    def knn_1_vs_rest(self):
        if self.normalized_path != None and self.lines[0].decision != None and len(self.lines[0].normalized) != 0:
            dialog_knn = DialogKnn1vsRest(
                self, self.config, self.lines, self.normalized_path)
        else:
            dialog_knn = DialogKnn1vsRest(self, self.config)

        dialog_knn.exec()

    def save_config_to_json(self):
        response = QFileDialog.getSaveFileName(
            parent=self,
            caption="Select a data file",
            directory="config.json"
        )

        if response[0] != '':
            with open(response[0], "w") as file:
                file.write(self.config.toJSON())

    def load_config_from_json(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a data file",
            directory=os.getcwd()
        )
        if response[0] != '':
            with open(response[0]) as file:
                cnf = json.load(file)
                self.config = normalization.Config(**cnf)

    def reset_config(self):
        self.config = None


class DialogKnn(QDialog, Ui_DialogKnn):
    def __init__(self, knn_type, parent=None, config=None, learning=[], learning_path=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connect_signals_slots()

        self.learning = learning
        self.test = []
        self.config = config
        self.learning_not_load = True
        self.knn_type = knn_type

        if len(learning) != 0:
            self.lineEditLearning.insert(learning_path)

        if self.knn_type == "one":
            self.setWindowTitle("Knn - typ pierwszy")
        elif self.knn_type == "two":
            self.setWindowTitle("Knn - typ drugi")

    def connect_signals_slots(self):
        self.pushButtonLoadLearning.clicked.connect(self.load_learning)
        self.pushButtonLoadTest.clicked.connect(self.load_test)
        self.pushButtonClassify.clicked.connect(self.classify)

    def load_learning(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a data file",
            directory=os.getcwd()
        )
        if response[0] != '':
            try:
                with open(response[0]) as file:
                    self.learning.clear()
                    splited_file = file.read().split("\n")

                    if self.lineEditSeparatorLearning.text() == "spacja":
                        separator = " "
                    else:
                        separator = self.lineEditSeparatorLearning.text()

                    for i in range(len(splited_file) - 1):
                        self.learning.append(normalization.Line(
                            splited_file[i], separator))

                    self.lineEditLearning.clear()

                    if self.lineEditTest.text() != "Błąd wczytywania" and self.lineEditTest != self.labelLearning.text():
                        self.pushButtonClassify.setEnabled(True)

                    self.lineEditLearning.insert(response[0])
            except:
                self.lineEditLearning.clear()
                self.lineEditLearning.insert("Błąd wczytywania")
                self.pushButtonClassify.setEnabled(False)

        self.learning_not_load = False

    def load_test(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a data file",
            directory=os.getcwd()
        )
        if response[0] != '':
            try:
                with open(response[0]) as file:
                    self.test.clear()
                    splited_file = file.read().split("\n")

                    if self.lineEditSeparatorTest.text() == "spacja":
                        separator = " "
                    else:
                        separator = self.lineEditSeparatorTest.text()

                    for i in range(len(splited_file) - 1):
                        self.test.append(normalization.Line(
                            splited_file[i], separator))

                    self.lineEditTest.clear()

                    if self.lineEditLearning.text() != "Błąd wczytywania" and self.lineEditLearning != self.labelTest.text():
                        self.pushButtonClassify.setEnabled(True)

                    self.lineEditTest.insert(response[0])

            except:
                self.lineEditTest.clear()
                self.lineEditTest.insert("Błąd wczytywania")
                self.pushButtonClassify.setEnabled(False)

    def classify(self):
        normalization.Line.Classify(self.test, self.learning, self.spinBoxK.value(
        ), self.comboBoxMetrics.currentText(), self.knn_type, self.learning_not_load, self.config)

        response = QFileDialog.getSaveFileName(
            parent=self,
            caption="Select a data file",
            directory="classified.dat"
        )

        if response[0] != '':
            with open(response[0], "w") as file:
                for line in self.test:
                    for i in range(len(line.values)):

                        if i == len(line.values) - 1:
                            file.write(str(line.normalized[i]))
                            break
                        if i == line.cnf.decisionID and line.decision != None:
                            file.write(str(line.decision) + ' ')
                            i -= 1
                            line.decision = None
                        else:
                            file.write(str(line.normalized[i]) + ' ')

                    if line.decision != None:
                        file.write(' ' + str(line.decision))
                    file.write("\n")


class DialogKnn1vsRest(QDialog, Ui_DialogKnn1vsRest):
    def __init__(self, parent=None, config=None, learning=[], learning_path=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connect_signals_slots()

        self.learning = learning
        self.config = config
        self.learning_not_load = True

        if len(learning) != 0:
            self.lineEditLearning.insert(learning_path)
            self.pushButtonClassify.setEnabled(True)

    def connect_signals_slots(self):
        self.pushButtonLoadLearning.clicked.connect(self.load_learning)
        self.pushButtonClassify.clicked.connect(self.classify)

    def load_learning(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a data file",
            directory=os.getcwd()
        )
        if response[0] != '':
            with open(response[0]) as file:
                self.learning.clear()
                splited_file = file.read().split("\n")

                if self.lineEditSeparatorLearning.text() == "spacja":
                    separator = " "
                else:
                    separator = self.lineEditSeparatorLearning.text()

                for i in range(len(splited_file) - 1):
                    self.learning.append(normalization.Line(
                        splited_file[i], separator))

                self.lineEditLearning.clear()
                self.pushButtonClassify.setEnabled(True)
                self.lineEditLearning.insert(response[0])

        self.learning_not_load = False

    def classify(self):
        if self.learning_not_load == False and self.config != None:
            for line in self.learning:
                line.cnf = self.config

            normalization.Line.Normalization(
                self.learning, self.learning[0].cnf.mi, self.learning[0].cnf.ma)
        else:
            normalization.Line.SetDecision(self.learning, 1)
            normalization.Line.ConvertSymbolicToNumeric(
                normalization.Line, self.learning)
            normalization.Line.FindMinMaxInNumeric(self.learning)
            normalization.Line.Normalization(
                self.learning, 0.0, 1.0)

        for line in self.learning:
            try:
                line.normalized.pop(line.cnf.decisionID)
            except:
                pass

        distance = {}
        coverage = 0
        accuracy = 0

        if self.comboBoxMetrics.currentText() == "Euklidesa":
            for one in self.learning:
                if len(np.array(list(one.normalized.values()))) == 0:
                    break
                for rest in self.learning:
                    if rest != one:
                        distance[rest] = dis.euclidean(
                            np.array(list(rest.normalized.values())), np.array(list(one.normalized.values())))
                if self.comboBoxKnn.currentText() == "Pierwszy":
                    decision = normalization.Line.GiveDecisionKnnOne(
                        distance, self.spinBoxK.value())
                elif self.comboBoxKnn.currentText() == "Drugi":
                    decision = normalization.Line.GiveDecisionKnnTwo(
                        distance, self.spinBoxK.value())
                if decision != "NaN":
                    coverage += 1
                    if decision == one.decision:
                        accuracy += 1

        elif self.comboBoxMetrics.currentText() == "Manhattan":
            for one in self.learning:
                if len(np.array(list(one.normalized.values()))) == 0:
                    break
                for rest in self.learning:
                    if rest != one:
                        distance[rest] = dis.euclidean(
                            np.array(list(rest.normalized.values())), np.array(list(one.normalized.values())))
                if self.comboBoxKnn.currentText() == "Pierwszy":
                    decision = normalization.Line.GiveDecisionKnnOne(
                        distance, self.spinBoxK.value())
                elif self.comboBoxKnn.currentText() == "Drugi":
                    decision = normalization.Line.GiveDecisionKnnTwo(
                        distance, self.spinBoxK.value())
                if decision != "NaN":
                    coverage += 1
                    if decision == one.decision:
                        accuracy += 1

        elif self.comboBoxMetrics.currentText() == "Canberra":
            for one in self.learning:
                if len(np.array(list(one.normalized.values()))) == 0:
                    break
                for rest in self.learning:
                    if rest != one:
                        distance[rest] = dis.euclidean(
                            np.array(list(rest.normalized.values())), np.array(list(one.normalized.values())))
                if self.comboBoxKnn.currentText() == "Pierwszy":
                    decision = normalization.Line.GiveDecisionKnnOne(
                        distance, self.spinBoxK.value())
                elif self.comboBoxKnn.currentText() == "Drugi":
                    decision = normalization.Line.GiveDecisionKnnTwo(
                        distance, self.spinBoxK.value())
                if decision != "NaN":
                    coverage += 1
                    if decision == one.decision:
                        accuracy += 1

        self.lineEditCoverage.clear()
        self.lineEditAccuracy.clear()
        self.lineEditCoverage.insert(
            str((coverage/len(self.learning))*100))
        self.lineEditAccuracy.insert(
            str((accuracy/len(self.learning))*100))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
