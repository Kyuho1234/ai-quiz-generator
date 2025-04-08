'use client';

import { useState } from 'react';
import { Box, Container, Tab, TabList, TabPanel, TabPanels, Tabs, Text } from '@chakra-ui/react';
import LoginButton from '@/components/LoginButton';
import SignUpButton from '@/components/SignUpButton';

export default function SignIn() {
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <Container maxW="container.sm" py={10}>
      <Box p={8} borderWidth={1} borderRadius="lg" boxShadow="lg">
        <Text fontSize="2xl" fontWeight="bold" mb={6} textAlign="center">
          AI 퀴즈 생성기
        </Text>
        <Tabs isFitted index={tabIndex} onChange={setTabIndex}>
          <TabList mb={4}>
            <Tab>로그인</Tab>
            <Tab>회원가입</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <LoginButton />
            </TabPanel>
            <TabPanel>
              <SignUpButton />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Container>
  );
}