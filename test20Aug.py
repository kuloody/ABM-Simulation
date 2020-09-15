def redStateCange(self):
        count, red, yellow, green = self.neighbour()
        if self.disruptiveTend > compute_ave_disruptive(self.model) and self.model.quality <= self.agent_state and self.behave > 5:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return

        if self.model.hyper_Impulsive == 1 and self.model.control < self.agent_state and self.behave_2 > 5:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return

        if self.model.hyper_Impulsive == 0 and self.model.control < self.agent_state:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return

        if red > 5 and self.type == 2:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return

    def yellowStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.disruptiveTend >= compute_ave_disruptive(self.model) and self.model.quality <= self.agent_state and self.behave <= 5:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return

        if  self.model.quality > self.agent_state and self.behave >= 5:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return

        # At random if control is less than student state
        if self.model.control < self.agent_state and self.type == 1:
            self.type = 2
            self.model.learning -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return
        # At general if control is high turn into passive
        if self.model.control > self.agent_state and self.type == 3:
            self.type = 2
            self.model.distruptive -= 1
            self.disrubted += 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return

        if red > 2 and self.type == 1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return

        # Change state based on majority of neighbours' color and agent's current color state

    def greenStateCange(self):

        count, red, yellow, green = self.neighbour()

        if self.disruptiveTend > compute_ave_disruptive(self.model) and self.model.quality > self.agent_state and self.behave < 5:
            if self.type <= 2:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return

        elif self.disruptiveTend <= compute_ave_disruptive(self.model) and self.model.quality > self.agent_state and self.behave >= 5 and self.type <= 2:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return

        elif self.model.hyper_Impulsive == 1 and self.model.control > self.agent_state and self.behave_2 < 5:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return

        elif green > 5 and self.type == 2:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            return

    def neighbourState(self):
        count, red, yellow, green = self.neighbour()
        # calculate the probability of each colour
        Pturn_red = red / count
        Pturn_green = green / count
        Pturn_yellow = yellow / count

        if self.type == 3:
            Pturn_red += 0.2
        elif self.type == 2:
            Pturn_yellow += 0.2
        else:
            Pturn_green += 0.2
        colour = max(Pturn_red, Pturn_green, Pturn_yellow)
        if Pturn_red == colour:
            self.type = 3
            return
        if Pturn_yellow == colour:
            self.type = 2
            return
        if Pturn_green == colour:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            return

    def changeState(self):
        # Change to red if inattentiveness score is high and teaching quality is low and state is passive for long
        if self.behave > 5 and self.model.control < 3 < self.yellowState:
            self.type = 3
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return
        # Change to red if hyber impulsive score is high and teaching quality is low and state is passive for long
        if self.behave_2 > 5 and self.model.quality < 3 < self.yellowState:
            self.type = 3
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return
        # Change to attentive (green) is hyber impulsive score is low and teaching quality is high and state is passive for long
        # if self.behave_2 < 5 and self.model.quality > 3 and self.yellowState > 3:
        if self.yellowState > 3:
            self.type = 1
            self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState = 1
            self.model.learning += 1
            self.set_start_math()
            return
        # Change to green if passive for long

        # Change to passive (yellow) if inattentiveness score is high and teaching control is low and state is green for long
        if self.behave > 5 and self.model.control < 3 and self.greenState > self.model.AttentionSpan:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return
        # ##Change to passive (yellow) if inattentiveness score is high and teaching quality is low and state is green for long
        if self.behave_2 > 5 and self.model.quality < 3 and self.greenState > self.model.AttentionSpan:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return
        # Change to passive (yellow) if inattentiveness score is high and teaching control is high and state is green for long
        # Student will lose interest if inattentiveness score is high regardless of teaching quality
        if self.behave > 5 and self.model.quality > 3 and self.greenState > self.model.AttentionSpan:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return
        # Change to passive (yellow) if hyber impulsive score is high and teaching control is high and state is green for long
        # Student will lose focus if hyber impulsive score is high regardless of teaching control
        if self.behave_2 > 5 and self.model.control > 3 and self.greenState > self.model.AttentionSpan:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return
        # Change to yellow if inattentiveness score is low
        if self.behave < 5 and self.model.control < 3 and self.redState > 2:
            self.type = 2
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return
        # Change to yellow if hyber impulsive score is low
        if self.behave_2 < 5 and self.model.quality < 3 and self.redState > 2:
            self.type = 2
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return
