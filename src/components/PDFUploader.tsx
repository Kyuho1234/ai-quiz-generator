'use client';

import { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  useToast,
  Progress,
  Text,
} from '@chakra-ui/react';
import axios from 'axios';

interface PDFUploaderProps {
  onQuestionsGenerated: (data: { questions: any[]; context: string }) => void;
}

export default function PDFUploader({ onQuestionsGenerated }: PDFUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const toast = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
      } else {
        toast({
          title: '에러',
          description: 'PDF 파일만 업로드 가능합니다.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast({
        title: '에러',
        description: 'PDF 파일을 선택해주세요.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsUploading(true);
      setUploadProgress(0);

      const response = await axios.post('http://localhost:8000/api/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            setUploadProgress(progress);
          }
        },
      });

      onQuestionsGenerated(response.data);

      toast({
        title: '성공',
        description: '문제가 생성되었습니다.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: '에러',
        description: '파일 업로드 중 오류가 발생했습니다.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <Box p={6} borderWidth={1} borderRadius="lg" width="100%">
      <VStack spacing={4}>
        <FormControl>
          <FormLabel>PDF 파일 업로드</FormLabel>
          <Input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={isUploading}
          />
        </FormControl>

        {isUploading && (
          <Box w="100%">
            <Text mb={2}>업로드 중... {Math.round(uploadProgress)}%</Text>
            <Progress value={uploadProgress} size="sm" colorScheme="blue" />
          </Box>
        )}

        <Button
          colorScheme="blue"
          onClick={handleUpload}
          isLoading={isUploading}
          loadingText="업로드 중..."
          width="100%"
        >
          업로드
        </Button>
      </VStack>
    </Box>
  );
}