import json
from scipy.spatial import distance as dis
import numpy as np


class Config:

    def __init__(self, min={}, max={}, mi=None, ma=None, decisionID=None, omittedAtributs=[]):

        self.min = min
        self.max = max

        self.mi = mi
        self.ma = ma
        self.decisionID = decisionID

        self.omittedAtributs = omittedAtributs

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Line:

    def __init__(self, line, separator=" "):

        self.cnf = Config()

        self.values = line.split(separator)

        self.decision = None

        self.numeric = {}
        self.numericFromSymbolic = {}

        self.symbolic = {}

        self.normalized = {}

        for i in range(0, len(self.values)):
            try:
                self.numeric[i] = float(
                    self.values[i].replace(".", ","))
            except:
                try:
                    self.numeric[i] = float(
                        self.values[i].replace(",", "."))
                except:
                    self.symbolic[i] = self.values[i]

    def SetDecision(lines, decisionID):
        for line in lines:
            try:
                INTdecisionID = int(decisionID) - 1
                if INTdecisionID >= 0:
                    line.decision = line.values[INTdecisionID]
                    line.cnf.decisionID = INTdecisionID
            except:
                pass

    def Normalization(lines, mi, ma):
        for line in lines:
            line.cnf.mi = mi
            line.cnf.ma = ma
            line.Normalize()

    def Normalize(self):
        for key in self.numeric:
            if key in self.cnf.omittedAtributs:
                continue
            else:
                self.normalized[key] = self.NormalizationPattern(
                    self.cnf.min[key], self.cnf.max[key], self.cnf.mi, self.cnf.ma, self.numeric[key])

        for key in self.numericFromSymbolic:
            if key in self.cnf.omittedAtributs:
                continue
            else:
                self.normalized[key] = self.NormalizationPattern(
                    self.cnf.min[key], self.cnf.max[key], self.cnf.mi, self.cnf.ma, self.numericFromSymbolic[key])

    def NormalizationPattern(self, min, max, mi, ma, value):
        return (value - min) / ((max - min) * (ma - mi) + mi)

    def ConvertSymbolicToNumeric(self, lines):
        line = lines[0]
        for key in line.symbolic:
            frequency = Line.ValuesFrequency(lines, key)
            assignment = Line.AssigningNumbersToSymbols(frequency)

            line.cnf.Min[key] = assignment[0]
            line.cnf.Max[key] = assignment[-1]

            for i in range(0, len(lines)):
                lines[i].ChangeSymbolToNumber(assignment, key)

    def ValuesFrequency(lines, key):
        frequency = {}

        for line in lines:
            symbol = line.values[key]

            if frequency.__contains__(symbol) == True:
                frequency[symbol] += 1
            else:
                frequency[symbol] = 1

        return frequency

    def AssigningNumbersToSymbols(frequency):
        result = dict(sorted(frequency.items(), key=lambda kv: kv[1]))
        assignment = {}
        i = 0

        for key in result:
            assignment[i] = result[key]
            i += 1

        return assignment

    def ChangeSymbolToNumber(self, assignment, key):
        if self.numericFromSymbolic.__contains__(key) == False:
            self.numericFromSymbolic[key] = 0

        symbol = self.values[key]
        number = assignment[symbol]
        self.numericFromSymbolic[key] = number

    def FindMinMaxInNumeric(lines):
        numeric = lines[0].numeric

        for key in numeric:
            minimum = numeric[key]
            maximum = numeric[key]

            lostValues = []

            for i in range(1, len(lines)):
                if lines[i].numeric.__contains__(key) == True:
                    if lines[i].numeric[key] < minimum:
                        minimum = lines[i].numeric[key]

                    if lines[i].numeric[key] > maximum:
                        maximum = lines[i].numeric[key]
                else:
                    lostValues.append(lines[i])

            for item in lostValues:
                item.numeric[key] = minimum
                if item.symbolic.__contains__(key) == True:
                    del item.symbolic[key]

            for line in lines:
                line.cnf.min[key] = minimum
                line.cnf.max[key] = maximum

    def Classify(test, learning, k, metric, knn_type, learning_not_load=False, config=None):
        if learning_not_load == True and config == None:
            for line in test:
                line.cnf = learning[0].cnf

            Line.Normalization(
                test, test[0].cnf.mi, test[0].cnf.ma)
        elif config != None:
            for line in test:
                line.cnf = config

            Line.Normalization(
                test, test[0].cnf.mi, test[0].cnf.ma)

            if learning_not_load == False:
                Line.Normalization(
                    learning, test[0].cnf.mi, test[0].cnf.ma)
        else:
            Line.SetDecision(learning, 1)
            Line.ConvertSymbolicToNumeric(
                Line, learning)
            Line.FindMinMaxInNumeric(learning)
            Line.Normalization(
                learning, 0.0, 1.0)

            for line in test:
                line.cnf = learning[0].cnf

            Line.Normalization(
                test, test[0].cnf.mi, test[0].cnf.ma)

        for line in learning:
            try:
                line.normalized.pop(line.cnf.decisionID)
            except:
                pass

        distance = {}
        if metric == "Euklidesa":
            for line in test:
                if len(np.array(list(line.normalized.values()))) == 0:
                    break
                for sample in learning:
                    distance[sample] = dis.euclidean(
                        np.array(list(sample.normalized.values())), np.array(list(line.normalized.values())))
                if knn_type == "one":
                    line.decision = Line.GiveDecisionKnnOne(
                        distance, k)
                elif knn_type == "two":
                    line.decision = Line.GiveDecisionKnnTwo(
                        distance, k)
        elif metric == "Manhattan":
            for line in test:
                if len(np.array(list(line.normalized.values()))) == 0:
                    break
                for sample in learning:
                    distance[sample] = dis.cityblock(
                        sample.normalized.values(), np.array(list(line.normalized.values())))
                if knn_type == "one":
                    line.decision = Line.GiveDecisionKnnOne(
                        distance, k)
                elif knn_type == "two":
                    line.decision = Line.GiveDecisionKnnTwo(
                        distance, k)
        elif metric == "Canberra":
            for line in test:
                if len(np.array(list(line.normalized.values()))) == 0:
                    break
                for sample in learning:
                    distance[sample] = dis.canberra(
                        sample.normalized.values(), np.array(list(line.normalized.values())))
                if knn_type == "one":
                    line.decision = Line.GiveDecisionKnnOne(
                        distance, k)
                elif knn_type == "two":
                    line.decision = Line.GiveDecisionKnnTwo(
                        distance, k)

    def GiveDecisionKnnOne(distance, k):
        distance = dict(sorted(distance.items(), key=lambda item: item[1]))
        keys = list(distance.keys())
        decisions = {}

        for i in range(0, k):
            if decisions.__contains__(keys[i].decision) == False and keys[i].decision != '':
                decisions[keys[i].decision] = 1
            elif keys[i].decision != '':
                decisions[keys[i].decision] += 1

        decisions = dict(sorted(decisions.items(), key=lambda item: item[1]))
        decisions_frequency = list(distance.values())
        if len(decisions_frequency) == 1 or decisions_frequency[-1] > decisions_frequency[-2]:
            return list(decisions.keys())[-1]
        else:
            return "NaN"

    def GiveDecisionKnnTwo(distance, k):
        distance = dict(sorted(distance.items(), key=lambda item: item[1]))
        keys = list(distance.keys())
        decisions_distance = {}
        decisions_frequency = {}

        for key in keys:
            if decisions_distance.__contains__(key.decision) == False and key.decision != '':
                decisions_distance[key.decision] = distance[key]
                decisions_frequency[key.decision] = 1
            elif key.decision != '' and decisions_frequency[key.decision] < k:
                decisions_distance[key.decision] += distance[key]
                decisions_frequency[key.decision] += 1

        decisions_distance = dict(
            sorted(decisions_distance.items(), key=lambda item: item[1]))
        decisions_frequency = list(distance.values())
        if len(list(decisions_distance.values())) == 1 or list(decisions_distance.values())[0] < list(decisions_distance.values())[1]:
            return list(decisions_distance.keys())[0]
        else:
            return "NaN"
