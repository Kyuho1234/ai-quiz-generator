'use client';

import React, { useState } from 'react';
import { Button, Input, VStack, FormControl, FormLabel, useToast } from '@chakra-ui/react';
import { signIn, signOut, useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';

export default function LoginButton() {
  const { data: session } = useSession();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      const result = await signIn('credentials', {
        email,
        password,
        redirect: false,
      });

      if (result?.error) {
        toast({
          title: '로그인 실패',
          description: '이메일이나 비밀번호를 확인해주세요.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: '로그인 성공',
          description: '환영합니다!',
          status: 'success',
          duration: 2000,
          isClosable: true,
        });
        router.push('/');
      }
    } catch (error) {
      toast({
        title: '오류 발생',
        description: '로그인 중 문제가 발생했습니다.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (session) {
    return (
      <VStack>
        <span>{session.user?.name}님 환영합니다</span>
        <Button 
          onClick={() => {
            signOut({ redirect: false });
            router.push('/auth/signin');
          }}
        >
          로그아웃
        </Button>
      </VStack>
    );
  }

  return (
    <form onSubmit={handleLogin}>
      <VStack spacing={4} align="stretch">
        <FormControl>
          <FormLabel>이메일</FormLabel>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="test@test.com"
            isDisabled={isLoading}
          />
        </FormControl>
        <FormControl>
          <FormLabel>비밀번호</FormLabel>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="1234"
            isDisabled={isLoading}
          />
        </FormControl>
        <Button 
          type="submit" 
          colorScheme="blue"
          isLoading={isLoading}
          loadingText="로그인 중..."
        >
          로그인
        </Button>
      </VStack>
    </form>
  );
}