'use client';

import React, { useState } from 'react';
import {
  Box,
  Button,
  Radio,
  RadioGroup,
  Stack,
  Text,
  Progress,
  Input,
  VStack,
  Heading,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useToast,
} from '@chakra-ui/react';
import axios from 'axios';

interface Question {
  question: string;
  correct_answer: string;
  explanation: string;
  options?: string[];
  type: 'multiple_choice' | 'short_answer';
}

interface QuizSectionProps {
  questions: Question[];
}

interface QuizResult {
  is_correct: boolean;
  feedback: string;
  question: string;
  user_answer: string;
  correct_answer: string;
}

interface QuizResults {
  results: QuizResult[];
  total_score: number;
  total_questions: number;
  score_percentage: number;
  overall_feedback: string;
}

export default function QuizSection({ questions }: QuizSectionProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<string[]>(new Array(questions.length).fill(''));
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<QuizResults | null>(null);
  const toast = useToast();

  const currentQuestion = questions[currentIndex];
  const progress = ((currentIndex + 1) / questions.length) * 100;

  const handleAnswerChange = (value: string) => {
    const newAnswers = [...answers];
    newAnswers[currentIndex] = value;
    setAnswers(newAnswers);
  };

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleSubmitAll = async () => {
    // 모든 문제에 답변했는지 확인
    const emptyAnswers = answers.some(answer => !answer.trim());
    if (emptyAnswers) {
      toast({
        title: '모든 문제에 답해주세요.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      setLoading(true);
      
      const answersData = questions.map((question, index) => ({
        question: question.question,
        user_answer: answers[index],
        correct_answer: question.correct_answer,
        question_type: question.type,
      }));

      const response = await axios.post<QuizResults>('http://localhost:8000/api/check-answers', {
        answers: answersData,
      });

      setResults(response.data);
      setSubmitted(true);
    } catch (error) {
      console.error('Error submitting answers:', error);
      toast({
        title: '제출 중 오류가 발생했습니다.',
        description: '잠시 후 다시 시도해주세요.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setCurrentIndex(0);
    setAnswers(new Array(questions.length).fill(''));
    setSubmitted(false);
    setResults(null);
  };

  if (submitted && results) {
    return (
      <VStack spacing={6} align="stretch" w="100%" p={4}>
        <Heading size="lg">퀴즈 결과</Heading>
        
        <Alert
          status={results.score_percentage >= 70 ? 'success' : 'warning'}
          variant="subtle"
          flexDirection="column"
          alignItems="flex-start"
          p={4}
          borderRadius="md"
        >
          <AlertTitle mb={2}>
            점수: {results.score_percentage.toFixed(1)}% ({results.total_score}/{results.total_questions})
          </AlertTitle>
          <AlertDescription whiteSpace="pre-wrap">
            {results.overall_feedback}
          </AlertDescription>
        </Alert>

        <VStack spacing={4} align="stretch">
          {results.results.map((result, index) => (
            <Box key={index} p={4} borderWidth={1} borderRadius="md">
              <Text fontWeight="bold" mb={2}>
                문제 {index + 1}: {result.question}
              </Text>
              <Text color={result.is_correct ? 'green.500' : 'red.500'} mb={2}>
                내 답변: {result.user_answer}
              </Text>
              <Text color="gray.600" mb={2}>
                정답: {result.correct_answer}
              </Text>
              <Text whiteSpace="pre-wrap">{result.feedback}</Text>
            </Box>
          ))}
        </VStack>

        <Button colorScheme="blue" onClick={handleReset}>
          다시 풀기
        </Button>
      </VStack>
    );
  }

  if (!currentQuestion) {
    return null;
  }

  return (
    <Box w="full">
      <VStack spacing={6} align="stretch">
        <Progress value={progress} size="sm" colorScheme="blue" />
        <Text>
          문제 {currentIndex + 1} / {questions.length}
        </Text>

        <Box p={4} borderWidth={1} borderRadius="md">
          <Text fontWeight="bold" mb={4}>
            {currentQuestion.question}
          </Text>

          {currentQuestion.type === 'multiple_choice' ? (
            <RadioGroup
              value={answers[currentIndex]}
              onChange={handleAnswerChange}
            >
              <Stack spacing={2}>
                {currentQuestion.options?.map((option, index) => (
                  <Radio key={index} value={option}>
                    {option}
                  </Radio>
                ))}
              </Stack>
            </RadioGroup>
          ) : (
            <Input
              value={answers[currentIndex]}
              onChange={(e) => handleAnswerChange(e.target.value)}
              placeholder="답을 입력하세요"
            />
          )}
        </Box>

        <Stack direction="row" spacing={4} justify="space-between">
          <Button
            onClick={handlePrevious}
            isDisabled={currentIndex === 0}
            colorScheme="gray"
          >
            이전
          </Button>
          {currentIndex === questions.length - 1 ? (
            <Button
              onClick={handleSubmitAll}
              colorScheme="blue"
              isLoading={loading}
            >
              제출하기
            </Button>
          ) : (
            <Button
              onClick={handleNext}
              colorScheme="blue"
              isDisabled={!answers[currentIndex]}
            >
              다음
            </Button>
          )}
        </Stack>
      </VStack>
    </Box>
  );
}