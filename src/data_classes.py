from typing import NamedTuple, Optional, List, Union
from datetime import datetime

class EediItem(NamedTuple):
    # Core identification fields
    question_id: int
    user_id: int
    quiz_session_id: float
    quiz_id: int
    
    # Answer-related fields
    is_correct: bool
    answer_value: int
    date_answered: datetime
    answer_type: str
    correct_answer: float
    
    # Question content fields
    construct_id: float
    construct_name: str
    question_text: str
    
    # Answer text fields
    answer_a_text: str
    answer_b_text: str
    answer_c_text: str
    answer_d_text: str
    
    # Explanation fields
    explanation_a: str
    explanation_b: str
    explanation_c: str
    explanation_d: str
    
    # Process field
    process: Union[str, List[str]]
    
    # Optional embedding fields (based on your code sample)
    question_embedding: Optional[List[float]] = None
    process_embedding: Optional[List[float]] = None

    @classmethod
    def from_interaction(cls, interaction):
        """
        Create an EediItem from an interaction (DataFrame row or similar)
        """
        return cls(
            question_id=interaction.QuestionId,
            user_id=interaction.UserId,
            quiz_session_id=interaction.QuizSessionId,
            quiz_id=interaction.QuizId,
            is_correct=interaction.IsCorrect,
            answer_value=interaction.AnswerValue,
            date_answered=interaction.DateAnswered,
            answer_type=interaction.AnswerType,
            correct_answer=interaction.CorrectAnswer,
            construct_id=interaction.ConstructId,
            construct_name=interaction.ConstructName,
            question_text=interaction.QuestionText,
            answer_a_text=interaction.AnswerAText,
            answer_b_text=interaction.AnswerBText,
            answer_c_text=interaction.AnswerCText,
            answer_d_text=interaction.AnswerDText,
            explanation_a=interaction.ExplanationA,
            explanation_b=interaction.ExplanationB,
            explanation_c=interaction.ExplanationC,
            explanation_d=interaction.ExplanationD,
            process=interaction.process,
            question_embedding=getattr(interaction, 'QuestionEmbedding', None),
            process_embedding=getattr(interaction, 'processEmbedding', None)
        )