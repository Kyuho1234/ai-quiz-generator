'use client';

import { ChakraProvider } from '@chakra-ui/react';
import { SessionProvider } from 'next-auth/react';

export default function Providers({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <SessionProvider refetchInterval={0}>
      <ChakraProvider>
        {children}
      </ChakraProvider>
    </SessionProvider>
  );
}