from abc import ABC, abstractmethod

class Judge(ABC):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    @abstractmethod
     # paramters: env state of last round, list of lables of last round, action of the agent last round
    def judgement(self, env, action): 
        pass

class JudgePrioR(Judge):
    def __init__(self, translator):
        super().__init__(translator=translator)

    def judgement(self, env, action):

        if 'D' in env.get_lables():
            if action in self.translator.impermissible('R', env):
                return (('D', 'R'))

        elif 'B' in env.get_lables():
            if action in self.translator.impermissible('C', env):
                return (('B', 'C'))

        return None
    
class JudgePrioW(Judge):
    def __init__(self, translator):
        super().__init__(translator=translator)
    
    def judgement(self, env, action): 

        if 'B' in env.get_lables():
            if action in self.translator.impermissible('C', env):
                return (('B', 'C'))
            
        elif 'D' in env.get_lables():
            if action in self.translator.impermissible('R', env):
                return (('D', 'R'))

        return None
    

class JudgeOnlyR(Judge):
    def __init__(self, translator):
        super().__init__(translator=translator)

    def judgement(self, env, action): 

        if 'D' in env.get_lables():
            if action in self.translator.impermissible('R', env):
                return (('D', 'R'))

        return None
    
class JudgeOnlyW(Judge):
    def __init__(self, translator):
        super().__init__(translator=translator)
    
    def judgement(self, env, action): 

        if 'B' in env.get_lables():
            if action in self.translator.impermissible('C', env):
                return (('B', 'C'))

        return None
